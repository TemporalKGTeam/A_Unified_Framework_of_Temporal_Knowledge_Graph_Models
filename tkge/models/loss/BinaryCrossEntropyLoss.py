from tkge.models.loss import Loss
from tkge.common.config import Config
from tkge.common.paramtype import *

import torch


@Loss.register(name="binary_cross_entropy_loss")
class BinaryCrossEntropyLoss(Loss):
    device = DeviceParam(name='device', default_value='cuda')

    def __init__(self, config: Config):
        super().__init__(config)

        self.device = config.get("task.device")

        self._loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, scores, labels, **kwargs):
        """Computes the loss given the scores and corresponding labels.

        `scores` is a batch_size x triples matrix holding the scores predicted by some
        model.

        `labels` is either (i) a batch_size x triples Boolean matrix holding the
        corresponding labels or (ii) a vector of positions of the (then unique) 1-labels
        for each row of `scores`.

        """

        return self._loss(scores, labels)
