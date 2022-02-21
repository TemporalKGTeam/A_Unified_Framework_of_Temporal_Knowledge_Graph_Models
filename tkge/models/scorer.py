import torch
from torch import nn

from typing import Type, Callable, Dict
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer


class BaseScorer(nn.Module):
    def __init__(self):
        super(BaseScorer, self).__init__()


class TemporalScorer(BaseScorer):
    def __init__(self):
        super(TemporalScorer, self).__init__()


