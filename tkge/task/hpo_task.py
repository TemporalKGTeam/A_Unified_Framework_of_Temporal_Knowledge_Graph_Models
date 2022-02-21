import os
import argparse

import ax
import random
import copy

from ax import Models
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

from tkge.task.task import Task
from tkge.task.train_task import TrainTask
from tkge.common.config import Config

from typing import Dict, Tuple


class HPOTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Hyperparameter optimization"""
        subparser = parser.add_parser("hpo", description=description, help="search hyperparameter.")

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="specify configuration file path"
        )

        return subparser

    def __init__(self, config: Config):
        super(HPOTask, self).__init__(config=config)

        self._prepare_experiment()

    def _prepare_experiment(self):
        gs = GenerationStrategy(name="sobol+gpei", steps=[
            GenerationStep(model=ax.Models.SOBOL,
                           num_trials=self.config.get("hpo.num_random_trials"),
                           min_trials_observed=self.config.get("hpo.num_random_trials") // 2,
                           enforce_num_trials=True),
            GenerationStep(model=ax.Models.GPEI, num_trials=-1)])

        # initialize a client
        self.ax_client = AxClient(generation_strategy=gs)

        # define the search space
        hp_group = self.config.get("hpo.hyperparam")

        # generation strategy
        # generation_strategy = GenerationStrategy(
        #     name="Sobol+GPEI",
        #     steps=[
        #         GenerationStep(
        #             model=Models.SOBOL,
        #             num_trials=self.config.get("hpo.num_trials"),
        #             min_trials_observed=ceil(self.config.get("hpo.num_trials") / 2),
        #             enforce_num_trials=True,
        #             model_kwargs={"seed": self.config.get("hpo.sobol_seed")},
        #         ),
        #         GenerationStep(model=Models.GPEI, num_trials=-1, max_parallelism=3),
        #     ],
        # )

        self.ax_client.create_experiment(
            name="hyperparam_search",
            parameters=hp_group,
            objective_name="mrr",
            minimize=False,
        )

    def _evaluate(self, parameters, trial_id) -> Dict[str, Tuple[float, float]]:
        """
        evaluate a trial given parameters and return the metrics
        """

        self.config.log(f"Start trial {trial_id}")
        self.config.log(f"with parameters {parameters}")

        # overwrite the config
        trial_config: Config = copy.deepcopy(self.config)
        for k, v in parameters.items():
            trial_config.set(k, v)

        trial_config.create_trial(trial_id)

        # initialize a trainer
        trial_trainer: TrainTask = TrainTask(trial_config)

        # train
        trial_trainer.main()
        best_metric = trial_trainer.best_metric

        self.config.log(f"End trial {trial_id}")
        self.config.log(f"best metric achieved at {best_metric}")

        # evaluate
        return {"mrr": (best_metric, 0.0)}

    def main(self):
        # generate trials/arms
        for i in range(self.config.get("hpo.num_trials")):
            parameters, trial_index = self.ax_client.get_next_trial()

            try:
                data = self._evaluate(parameters, trial_index)
            except Exception as e:
                self.config.log(str(e), level="warning")
                self.ax_client.log_trial_failure(trial_index=trial_index)
            else:
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=data)

        best_parameters, values = self.ax_client.get_best_parameters()

        self.config.log("Search task finished.")
        self.config.log(f"Best parameter:"
                        f"{best_parameters}"
                        f""
                        f"Best metrics:"
                        f"{values}")

        if hasattr(self.ax_client.generation_strategy, 'trials_as_df'):
            self.ax_client.generation_strategy.trials_as_df.to_csv(self.config.ex_folder + '/trials.csv')
        self.ax_client.save_to_json_file(filepath=self.config.ex_folder + '/trials.json')
