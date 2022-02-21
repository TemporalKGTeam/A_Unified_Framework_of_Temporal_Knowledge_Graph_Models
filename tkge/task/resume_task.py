import torch

import time
import os
import argparse

from typing import Dict, List
from collections import defaultdict

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.train.optim import get_optimizer, get_scheduler
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.pipeline_model import PipelineModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation


class ResumeTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Resume training"""
        subparser = parser.add_parser("resume", description=description, help="resume previous experiments.")

        subparser.add_argument(
            "-e",
            "--ex",
            type=str,
            help="specify the experiment folder",
            dest='experiment'
        )

        subparser.add_argument(
            "--overrides",
            action="store_true",
            default=False,
            help="override the hyper-parameter stored in checkpoint with a new configuration file"
        )

        subparser.add_argument(
            "--checkpoint",
            type=str,
            default="latest.ckpt",
            dest="ckpt_name",
            help="choose the checkpoint name in experiment folder from which training will be resumed."
        )

        return subparser

    def __init__(self, config: Config, ckpt_path: str):
        super().__init__(config=config)
        self.ckpt_path = ckpt_path

        self.dataset: DatasetProcessor = self.config.get("dataset.name")
        self.train_loader: torch.utils.data.DataLoader = None
        self.valid_loader: torch.utils.data.DataLoader = None
        # self.test_loader = None
        self.sampler: NegativeSampler = None
        self.model: BaseModel = None
        self.loss: Loss = None
        self.optimizer: torch.optim.optimizer.Optimizer = None
        self.lr_scheduler = None
        self.evaluation: Evaluation = None

        self.train_bs = self.config.get("train.batch_size")
        self.valid_bs = self.config.get("train.valid.batch_size")
        self.train_sub_bs = self.config.get("train.subbatch_size") if self.config.get(
            "train.subbatch_size") else self.train_bs
        self.valid_sub_bs = self.config.get("train.valid.subbatch_size") if self.config.get(
            "train.valid.subbatch_size") else self.valid_bs

        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self.ckpt = torch.load(ckpt_path)

        self._prepare()

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}...")
        self.dataset = DatasetProcessor.create(config=self.config)
        self.dataset.info()

        self.config.log(f"Loading training split data for loading")
        # TODO(gengyuan) load params
        self.train_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("train"), self.datatype),
            shuffle=True,
            batch_size=self.train_bs,
            num_workers=self.config.get("train.loader.num_workers"),
            pin_memory=self.config.get("train.loader.pin_memory"),
            drop_last=self.config.get("train.loader.drop_last"),
            timeout=self.config.get("train.loader.timeout")
        )

        self.valid_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("test"), self.datatype + ['timestamp_id']),
            shuffle=False,
            batch_size=self.valid_bs,
            num_workers=self.config.get("train.loader.num_workers"),
            pin_memory=self.config.get("train.loader.pin_memory"),
            drop_last=self.config.get("train.loader.drop_last"),
            timeout=self.config.get("train.loader.timeout")
        )

        self.config.log(f"Initializing negative sampling")
        self.sampler = NegativeSampler.create(config=self.config, dataset=self.dataset)
        self.onevsall_sampler = NonNegativeSampler(config=self.config, dataset=self.dataset, as_matrix=True)

        self.config.log(f"Loading model {self.config.get('model.type')} from {self.ckpt_path}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.load_state_dict(self.ckpt['state_dict'], strict=True)
        self.model.to(self.device)

        self.config.log(f"Initializing loss function")
        self.loss = Loss.create(config=self.config)

        self.config.log(f"Restoring optimizer")
        optimizer_type = self.config.get("train.optimizer.type")
        optimizer_args = self.config.get("train.optimizer.args")
        self.optimizer = get_optimizer(self.model.parameters(), optimizer_type, optimizer_args)
        self.optimizer.load_state_dict(self.ckpt['optimizer'])

        self.config.log(f"Restoring lr scheduler")
        if self.config.get("train.lr_scheduler"):
            scheduler_type = self.config.get("train.lr_scheduler.type")
            scheduler_args = self.config.get("train.lr_scheduler.args")
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_type, scheduler_args)
            self.lr_scheduler.load_state_dict(self.ckpt['lr_scheduler'])

        self.config.log(f"Initializing regularizer")
        self.regularizer = dict()
        self.inplace_regularizer = dict()

        if self.config.get("train.regularizer"):
            for name in self.config.get("train.regularizer"):
                self.regularizer[name] = Regularizer.create(self.config, name)

        if self.config.get("train.inplace_regularizer"):
            for name in self.config.get("train.inplace_regularizer"):
                self.inplace_regularizer[name] = InplaceRegularizer.create(self.config, name)

        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

        # validity checks and warnings
        self.subbatch_adaptive = self.config.get("train.subbatch_adaptive")

        if self.train_sub_bs >= self.train_bs or self.train_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.sub_batch_size={self.train_sub_bs} is greater or equal to "
                            f"train.batch_size={self.train_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.train_sub_bs = self.train_bs

        if self.valid_sub_bs >= self.valid_bs or self.valid_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.valid.sub_batch_size={self.valid_sub_bs} is greater or equal to "
                            f"train.valid.batch_size={self.valid_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.valid_sub_bs = self.valid_bs

    def main(self):
        self.config.log("RESUME TRAINING")

        save_freq = self.config.get("train.checkpoint.every")
        eval_freq = self.config.get("train.valid.every")

        self.best_metric = 0. if 'best_metrics' not in self.ckpt else self.ckpt['best_metrics']
        self.best_epoch = 0 if 'best_epoch' not in self.ckpt else self.ckpt['best_epoch']

        for epoch in range(self.ckpt['last_epoch'] + 1, self.config.get("train.max_epochs") + 1):
            self.model.train()

            total_epoch_loss = 0.0
            train_size = self.dataset.train_size

            start_time = time.time()

            # processing batches
            for pos_batch in self.train_loader:
                done = False

                while not done:
                    try:
                        self.optimizer.zero_grad()

                        batch_loss = 0.

                        # may be smaller than the specified batch size in last iteration
                        bs = pos_batch.size(0)

                        # processing subbatches
                        for start in range(0, bs, self.train_sub_bs):
                            stop = min(start + self.train_sub_bs, bs)
                            pos_subbatch = pos_batch[start:stop]
                            subbatch_loss, subbatch_factors = self._subbatch_forward(pos_subbatch)

                            batch_loss += subbatch_loss

                        batch_loss.backward()
                        self.optimizer.step()

                        total_epoch_loss += batch_loss.cpu().item()

                        if subbatch_factors:
                            for name, tensors in subbatch_factors.items():
                                if name not in self.inplace_regularizer:
                                    continue

                                if not isinstance(tensors, (tuple, list)):
                                    tensors = [tensors]

                                self.inplace_regularizer[name](tensors)

                        done = True

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.train_sub_bs //= 2
                        if self.train_sub_bs > 0:
                            self.config.log(f"CUDA out of memory. Subbatch size reduced to {self.train_sub_bs}.",
                                            level="warning")
                        else:
                            self.config.log(f"CUDA out of memory. Subbatch size cannot be further reduces.",
                                            level="error")
                            raise e

                # empty caches
                # del samples, labels, scores, factors
                # if self.device=="cuda":
                #     torch.cuda.empty_cache()

            stop_time = time.time()
            avg_loss = total_epoch_loss / train_size

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(avg_loss)
                else:
                    self.lr_scheduler.step()

            self.config.log(f"Loss in iteration {epoch} : {avg_loss} consuming {stop_time - start_time}s")

            if epoch % save_freq == 0:
                self.config.log(f"Save the model checkpoint to {self.config.checkpoint_folder} as file epoch_{epoch}.ckpt")
                self.save_ckpt(f"epoch_{epoch}", epoch=epoch)

            if epoch % eval_freq == 0:
                metrics = self.eval()

                self.config.log(f"Metrics(head prediction) in iteration {epoch} : {metrics['head'].items()}")
                self.config.log(f"Metrics(tail prediction) in iteration {epoch} : {metrics['tail'].items()}")
                self.config.log(f"Metrics(both prediction) in iteration {epoch} : {metrics['avg'].items()} ")

                if metrics['avg']['mean_reciprocal_ranking'] > self.best_metric:
                    self.best_metric = metrics['avg']['mean_reciprocal_ranking']
                    self.best_epoch = epoch

                    self.save_ckpt('best', epoch=epoch)
                else:
                    if self.config.get('train.valid.early_stopping.early_stop'):
                        patience = self.config.get('train.valid.early_stopping.patience')
                        if epoch - self.best_epoch >= patience:
                            self.config.log(
                                f"Early stopping: valid metrics not improved in {patience} epoch and training stopped at epoch {epoch}")
                            break

                if self.config.get('train.valid.early_stopping.early_stop'):
                    thresh_epoch = self.config.get('train.valid.early_stopping.epochs')
                    if epoch > thresh_epoch and self.best_metric < self.config.get(
                            'train.valid.early_stopping.metric_thresh'):
                        self.config.log(
                            f"Early stopping: within {thresh_epoch} metrics doesn't exceed threshold and training stopped at epoch {epoch}")
                        break

            self.save_ckpt('latest', epoch=epoch)

        self.config.log(f"TRAINING FINISHED: Best model achieved at epoch {self.best_epoch}")

    def _subbatch_forward(self, pos_subbatch):
        sample_target = self.config.get("negative_sampling.target")
        samples, labels = self.sampler.sample(pos_subbatch, sample_target)

        samples = samples.to(self.device)
        labels = labels.to(self.device)

        scores, factors = self.model.fit(samples)

        self.config.assert_true(scores.size(0) == labels.size(
            0), f"Score's size {scores.shape} should match label's size {labels.shape}")
        loss = self.loss(scores, labels)

        factors = {} if factors==None else factors
        self.config.assert_true(not (factors and set(factors.keys()) - (set(self.regularizer) | set(
            self.inplace_regularizer))),
                                f"Regularizer name defined in model {set(factors.keys())} should correspond to that in config file")

        if factors:
            for name, tensors in factors.items():
                if name not in self.regularizer:
                    continue

                if not isinstance(tensors, (tuple, list)):
                    tensors = [tensors]

                reg_loss = self.regularizer[name](tensors)
                loss += reg_loss

        return loss, factors

    def _subbatch_forward_predict(self, query_subbatch):
        bs = query_subbatch.size(0)
        queries_head = query_subbatch.clone()[:, :-1]
        queries_tail = query_subbatch.clone()[:, :-1]

        queries_head[:, 0] = float('nan')
        queries_tail[:, 2] = float('nan')

        batch_scores_head = self.model.predict(queries_head)
        self.config.assert_true(list(batch_scores_head.shape) == [bs,
                                                                  self.dataset.num_entities()],
                                f"Scores {batch_scores_head.shape} should be in shape [{bs}, {self.dataset.num_entities()}]")

        batch_scores_tail = self.model.predict(queries_tail)
        self.config.assert_true(list(batch_scores_tail.shape) == [bs,
                                                                  self.dataset.num_entities()],
                                f"Scores {batch_scores_head.shape} should be in shape [{bs}, {self.dataset.num_entities()}]")

        subbatch_metrics = dict()

        subbatch_metrics['head'] = self.evaluation.eval(query_subbatch, batch_scores_head, miss='s')
        subbatch_metrics['tail'] = self.evaluation.eval(query_subbatch, batch_scores_tail, miss='o')
        subbatch_metrics['size'] = bs

        del query_subbatch, batch_scores_head, batch_scores_tail
        torch.cuda.empty_cache()

        return subbatch_metrics

    def eval(self):
        with torch.no_grad():
            self.model.eval()

            counter = 0

            metrics = dict()
            metrics['head'] = defaultdict(float)
            metrics['tail'] = defaultdict(float)
            metrics['size'] = 0

            for batch in self.valid_loader:
                done = False

                while not done:
                    try:
                        bs = batch.size(0)
                        dim = batch.size(1)

                        batch_metrics = dict()
                        batch_metrics['head'] = defaultdict(float)
                        batch_metrics['tail'] = defaultdict(float)
                        batch_metrics['size'] = 0

                        batch = batch.to(self.device)

                        counter += bs

                        for start in range(0, bs, self.valid_sub_bs):
                            stop = min(start + self.valid_sub_bs, bs)
                            query_subbatch = batch[start:stop]
                            subbatch_metrics = self._subbatch_forward_predict(query_subbatch)

                            for pos in ['head', 'tail']:
                                for key in subbatch_metrics[pos].keys():
                                    batch_metrics[pos][key] += subbatch_metrics[pos][key] * subbatch_metrics['size']
                            batch_metrics['size'] += subbatch_metrics['size']

                        done = True

                        for pos in ['head', 'tail']:
                            for key in batch_metrics[pos].keys():
                                batch_metrics[pos][key] /= batch_metrics['size']

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.valid_sub_bs //= 2
                        if self.valid_sub_bs > 0:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation reduced to {self.valid_sub_bs}.",
                                level="warning")
                        else:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation cannot be further reduces.",
                                level="error")
                            raise e

                for pos in ['head', 'tail']:
                    for key in batch_metrics[pos].keys():
                        metrics[pos][key] += batch_metrics[pos][key] * batch_metrics['size']
                metrics['size'] += batch_metrics['size']

            del batch
            torch.cuda.empty_cache()

            for pos in ['head', 'tail']:
                for key in metrics[pos].keys():
                    metrics[pos][key] /= metrics['size']

            avg = {k: (metrics['head'][k] + metrics['tail'][k]) / 2 for k in metrics['head'].keys()}

            metrics.update({'avg': avg})

            return metrics

    def save_ckpt(self, ckpt_name, epoch):
        filename = f"{ckpt_name}.ckpt"

        checkpoint = {
            'last_epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_metrics': self.best_metric,
            'best_epoch': self.best_epoch
        }

        torch.save(checkpoint,
                   os.path.join(self.config.checkpoint_folder,
                                filename))  # os.path.join(model, dataset, folder, filename))
