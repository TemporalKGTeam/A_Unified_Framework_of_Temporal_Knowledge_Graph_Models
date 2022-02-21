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
from tkge.common.utils import LocalConfig
from tkge.models.model import BaseModel
from tkge.models.loss import Loss
from tkge.models.fusion import TemporalFusion
from tkge.models.transformation import Transformation
from tkge.eval.metrics import Evaluation

from typing import Dict


class SearchTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Search a model"""
        subparser = parser.add_parser("search", description=description, help="search a model architecture.")

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="specify configuration file path"
        )

        return subparser

    def __init__(self, config: Config):
        super(SearchTask, self).__init__(config=config)

        self.dataset = self.config.get("dataset.name")
        self.train_loader: torch.utils.data.DataLoader = None
        self.valid_loader: torch.utils.data.DataLoader = None

        self.sampler: NegativeSampler = None

        self.loss: Loss = None

        self.optimizer: torch.optim.optimizer.Optimizer = None
        self.lr_scheduler = None
        self.evaluation: Evaluation = None

        # self.train_bs = self.config.get("train.batch_size")
        # self.valid_bs = self.config.get("train.valid.batch_size")
        # self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
        #     ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        self.device = self.config.get("task.device")

    def _prepare(self):
        pass

    def main(self):
        fusion: Dict[str, Dict] = self.config.get("search.fusion")
        transformation: Dict[str, Dict] = self.config.get("search.transformation")

        # TODO(gengyuan) use a context manager for logging and model configuration
        for f_k, f_v in fusion.items():
            for t_k, t_v in transformation.items():

                with LocalConfig(self.config, fusion=f_k, transformation=t_k) as local_config:
                    # constrain the embedding space based on the fusion and transformation types
                    target = local_config.get('fusion.target')

                    in_fusion_constraints, out_fusion_constraints = TemporalFusion.by_name(f_k).embedding_constraints()
                    in_tf_constraints = Transformation.by_name(t_k).embedding_constraints()

                    # Currently only support temporal encoded to either ent or rel
                    if 'ent+temp' in target:
                        ent_keys = in_fusion_constraints['operand1']
                        temp_keys = in_fusion_constraints['operand2']

                        fused_ent_keys = out_fusion_constraints['result']

                        if fused_ent_keys != in_tf_constraints['entity']:
                            # end this loop
                            continue
                        rel_keys = in_tf_constraints['relation']

                    if 'rel+temp' in target:
                        rel_keys = in_fusion_constraints['operand1']
                        temp_keys = in_fusion_constraints['operand2']

                        fused_rel_keys = out_fusion_constraints['result']

                        if fused_rel_keys != in_tf_constraints['relation']:
                            # end this loop
                            continue
                        ent_keys = in_tf_constraints['entity']

                    # update the local config file:
                    # dataset
                    # embedding

                    # create a new pipeline model


                    # save the model and all the metadata

                    # close the model


    def dep__init__(self, config: Config):
        super(SearchTask, self).__init__(config=config)

        self.dataset = self.config.get("dataset.name")
        self.test_loader = None
        self.sampler = None
        self.model = None
        self.evaluation = None

        self.test_bs = self.config.get("test.batch_size")
        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self._prepare()

        self.test()

    def dep_prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}")
        self.dataset = DatasetProcessor.create(config=self.config)

        self.config.log(f"Loading testing split data for loading")
        # TODO(gengyuan) load params
        self.test_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("test"), self.datatype + ['timestamp_id']),
            shuffle=False,
            batch_size=self.test_bs,
            num_workers=self.config.get("test.loader.num_workers"),
            pin_memory=self.config.get("test.loader.pin_memory"),
            drop_last=self.config.get("test.loader.drop_last"),
            timeout=self.config.get("test.loader.timeout")
        )

        self.onevsall_sampler = NonNegativeSampler(config=self.config, dataset=self.dataset, as_matrix=True)

        self.config.log(f"Loading model {self.config.get('model.name')}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset, device=self.device)
        model_path = self.config.get("test.model_path")
        model_state_dict = torch.load(model_path)

        self.model.load_state_dict(model_state_dict['state_dict'])

        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

    def dep_test(self):
        self.config.log("BEGIN TESTING")

        with torch.no_grad():
            self.model.eval()

            l = 0

            metrics = dict()
            metrics['head'] = defaultdict(float)
            metrics['tail'] = defaultdict(float)

            for batch in self.test_loader:
                bs = batch.size(0)
                dim = batch.size(1)
                l += bs

                samples_head, _ = self.onevsall_sampler.sample(batch, "head")
                samples_tail, _ = self.onevsall_sampler.sample(batch, "tail")

                samples_head = samples_head.to(self.device)
                samples_tail = samples_tail.to(self.device)

                batch_scores_head, _ = self.model.predict(samples_head)
                batch_scores_tail, _ = self.model.predict(samples_tail)

                batch_metrics = dict()
                batch_metrics['head'] = self.evaluation.eval(batch, batch_scores_head, miss='s')
                batch_metrics['tail'] = self.evaluation.eval(batch, batch_scores_tail, miss='o')

                for pos in ['head', 'tail']:
                    for key in batch_metrics[pos].keys():
                        metrics[pos][key] += batch_metrics[pos][key] * bs

            for pos in ['head', 'tail']:
                for key in metrics[pos].keys():
                    metrics[pos][key] /= l

            self.config.log(f"Metrics(head prediction) : {metrics['head'].items()}")
            self.config.log(f"Metrics(tail prediction) : {metrics['tail'].items()}")
