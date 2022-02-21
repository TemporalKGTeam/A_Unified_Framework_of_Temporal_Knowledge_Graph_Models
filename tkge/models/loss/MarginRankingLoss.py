from tkge.models.loss import Loss
from tkge.common.config import Config
from tkge.common.paramtype import *

import torch


@Loss.register(name="margin_ranking_loss")
class MarginRankingLoss(Loss):
    device = DeviceParam('device', default_value='cuda')
    margin = NumberParam('margin', default_value=100)

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.margin = self.config.get("train.loss.margin")
        self.device = self.config.get("task.device")
        # self.num_samples = self.config.get("negative_sampling.num_samples")

        self._loss = torch.nn.MarginRankingLoss(margin=self.margin, **kwargs)

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor):
        self.config.assert_true(labels.dim() == 2, 'Margin ranking loss only supports matrix-like scores and scores. Set train.negative_sampling.as_matrix to True in configuration file.')

        bs = scores.size(0)
        ns = scores.size(1) - 1

        # walkaround: assume the 1st column are positive samples and others are negative

        positive_scores = scores[:, 0].reshape(-1, 1)
        negative_scores = scores[:, 1:]

        positive_scores = positive_scores.repeat((ns, 1)).squeeze()
        negative_scores = negative_scores.reshape(-1)

        y = torch.ones_like(positive_scores)

        return self._loss(positive_scores, negative_scores, y)

    # def __call__(self, scores, labels, **kwargs):
    #     # scores is (batch_size x num_negatives + 1)
    #     labels = self._labels_as_matrix(scores, labels)
    #
    #     if "negative_sampling" in self._train_type:
    #         # Pair each 1 with the following zeros until next 1
    #         labels = labels.to(self._device).view(-1)
    #         pos_positives = labels.nonzero().view(-1)
    #         pos_negatives = (labels == 0).nonzero().view(-1)
    #
    #         n_over_p = pos_negatives.size(0) // pos_positives.size(0)
    #         # repeat each positive score num_negatives times
    #         pos_positives = (
    #             pos_positives.view(-1, 1).repeat(1, n_over_p).view(-1)
    #         )
    #         positives = scores.view(-1)[pos_positives].to(self._device).view(-1)
    #         negatives = scores.view(-1)[pos_negatives].to(self._device).view(-1)
    #         target = torch.ones(positives.size()).to(self._device)
    #         return self._loss(positives, negatives, target)
    #
    #     elif self._train_type == "KvsAll":
    #         # TODO determine how to form pairs for margin ranking in KvsAll training
    #         # scores and labels are tensors of size (batch_size, num_entities)
    #         # Each row has 1s and 0s of a single sp or po tuple from training
    #         # How to combine them for pairs?
    #         # Each 1 with all 0s? Can memory handle this?
    #         raise NotImplementedError(
    #             "Margin ranking with KvsAll training not yet supported."
    #         )
    #     else:
    #         raise ValueError("train.type for margin ranking.")
