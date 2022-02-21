import torch

import time
import os
from collections import defaultdict
import argparse

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation


class EvaluateTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Eval a model"""
        subparser = parser.add_parser("eval", description=description, help="evaluate a model.")

        subparser.add_argument(
            "-e",
            "--ex",
            type=str,
            help="specify the experiment folder",
            dest='experiment'
        )

        subparser.add_argument(
            "--checkpoint",
            type=str,
            default="best.ckpt",
            dest="ckpt_name",
            help="choose the checkpoint name in experiment folder from which training will be resumed."
        )

        return subparser

    def __init__(self, config: Config, ckpt_path: str):
        super(EvaluateTask, self).__init__(config=config)
        self.ckpt_path = ckpt_path

        self.dataset = self.config.get("dataset.name")
        self.test_loader = None
        self.sampler = None
        self.model = None
        self.evaluation = None

        self.test_bs = self.config.get("train.valid.batch_size")
        self.test_sub_bs = self.config.get("train.valid.subbatch_size") if self.config.get(
            "train.valid.subbatch_size") else self.test_bs

        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self.ckpt = torch.load(ckpt_path)

        self._prepare()

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}")
        self.dataset = DatasetProcessor.create(config=self.config)

        self.config.log(f"Loading testing split data for loading")
        # TODO(gengyuan) load params
        self.test_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("test"), self.datatype + ['timestamp_id']),
            shuffle=False,
            batch_size=self.test_bs
        )

        self.config.log(f"Loading model {self.config.get('model.type')} from {self.ckpt_path}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.load_state_dict(self.ckpt['state_dict'], strict=True)
        self.model.to(self.device)


        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

        # validity checks and warnings
        self.subbatch_adaptive = self.config.get("train.subbatch_adaptive")

        if self.test_sub_bs >= self.test_bs or self.test_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.valid.sub_batch_size={self.test_sub_bs} is greater or equal to "
                            f"train.valid.batch_size={self.test_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.test_sub_bs = self.test_bs

    def main(self):
        self.config.log("BEGIN TESTING")

        metrics = self.eval()

        self.config.log(f"Metrics(head prediction): {metrics['head'].items()}")
        self.config.log(f"Metrics(tail prediction): {metrics['tail'].items()}")
        self.config.log(f"Metrics(both prediction): {metrics['avg'].items()} ")


    def eval(self):
        with torch.no_grad():
            self.model.eval()

            counter = 0

            metrics = dict()
            metrics['head'] = defaultdict(float)
            metrics['tail'] = defaultdict(float)
            metrics['size'] = 0

            for batch in self.test_loader:
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

                        for start in range(0, bs, self.test_sub_bs):
                            stop = min(start + self.test_sub_bs, bs)
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

                        self.test_sub_bs //= 2
                        if self.test_sub_bs > 0:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation reduced to {self.test_sub_bs}.",
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