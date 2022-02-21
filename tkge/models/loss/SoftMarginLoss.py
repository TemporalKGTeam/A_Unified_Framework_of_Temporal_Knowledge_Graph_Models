from tkge.models.loss import Loss

import torch


@Loss.register(name="soft_margin_loss")
class SoftMarginLoss(Loss):
    def __init__(self, config, reduction="sum", **kwargs):
        super().__init__(config)

        self._device = config.get("task.device")

        self._loss = torch.nn.SoftMarginLoss(reduction=reduction, **kwargs)

    def __call__(self, scores, labels, **kwargs):
        labels = self._labels_as_matrix(scores, labels)
        labels = labels * 2 - 1  # expects 1 / -1 as label

        return self._loss(scores.view(-1), labels.view(-1))
