from typing import Optional
import argparse

from tkge.common.configurable import Configurable
from tkge.common.registrable import Registrable
from tkge.common.config import Config
from tkge.data.dataset import DatasetProcessor


class Task:
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        raise NotImplementedError

    def __init__(self, config: Config):
        self.config = config

    # @staticmethod
    # def create(task_type: str,
    #            config: Config):
    #     if task_type in Task.list_available():
    #         return Task.by_name(task_type)(config)

    # @staticmethod
    # def create(
    #         config: Config,
    #         dataset: Optional[DatasetProcessor] = None,
    #         parent_job=None,
    #         model=None
    # ):
    #     "Create a new job."
    #     from kge.job import TrainingJob, EvaluationJob, SearchJob
    #
    #     if dataset is None:
    #         dataset = DatasetProcessor.create(config)
    #
    #     job_type = config.get("job.type")
    #     if job_type == "train":
    #         return TrainingJob.create(
    #             config, dataset, parent_job=parent_job, model=model
    #         )
    #     elif job_type == "search":
    #         return SearchJob.create(config, dataset, parent_job=parent_job)
    #     elif job_type == "eval":
    #         return EvaluationJob.create(
    #             config, dataset, parent_job=parent_job, model=model
    #         )
    #     else:
    #         raise ValueError("unknown job type")
