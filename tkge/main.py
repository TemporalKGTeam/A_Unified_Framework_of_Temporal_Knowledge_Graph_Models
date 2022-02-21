import argparse

from tkge.task.train_task import TrainTask
from tkge.common.config import Config


desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-config', help='configuration file folder', type=str)
args = parser.parse_args()

config = Config(folder=args.config)

trainer = TrainTask(config)
