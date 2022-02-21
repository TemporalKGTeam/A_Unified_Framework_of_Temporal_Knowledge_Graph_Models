from torch import nn
import torch
from torch.nn import functional as F

from abc import ABC, abstractmethod
from typing import Dict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.paramtype import *
from tkge.common.error import ConfigurationError


class Transformation(ABC, nn.Module, Registrable, Configurable):
    def __init__(self, config: Config):
        nn.Module.__init__(self)
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

    @classmethod
    def create_from_name(cls, config: Config):
        transformation_type = config.get("model.transformation.type")
        kwargs = config.get("model.transformation.args")

        kwargs = kwargs if not isinstance(kwargs, type(None)) else {}

        if transformation_type in Transformation.list_available():
            # kwargs = config.get("model.args")  # TODO: get all args params
            return Transformation.by_name(transformation_type)(config, **kwargs)
        else:
            raise ConfigurationError(
                f"{transformation_type} specified in configuration file is not supported "
                f"implement your model class with `Transformation.register(name)"
            )

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def embedding_constraint():
        raise NotImplementedError


@Transformation.register(name="translation_tf")
class TranslationTransformation(Transformation):
    gamma = NumberParam('gamma', default_value=100)
    p = NumberParam('p', default_value=1)

    def __init__(self, config):
        super(TranslationTransformation, self).__init__(config=config)

        self.gamma = self.config.get('model.transformation.gamma')
        self.p = self.config.get('model.transformation.p')

    def forward(self, head, rel, tail, summation=True):
        scores = head['real'] + rel['real'] - tail['real']
        if summation:
            scores = self.gamma - torch.norm(scores, p=self.p, dim=1)

        return scores

    @staticmethod
    def embedding_constraint():
        """
        Translation only support real embeddings.
        """
        constraints = {'entity': ['real'],
                       'relation': ['real']}

        return constraints


@Transformation.register(name="rotation_tf")
class RotationTransformation(Transformation):
    gamma = NumberParam('gamma', default_value=100)

    def __init__(self, config):
        super(RotationTransformation, self).__init__(config=config)

        self.gamma = self.config.get('model.transformation.gamma')
        self.range = self.config.get('model.transformation.range')

    def forward(self, head: Dict[str, torch.Tensor], rel: Dict[str, torch.Tensor], tail: Dict[str, torch.Tensor],
                summation=True):
        """
        head and tail should be Dict[str, torch.Tensor]
        """
        pi = 3.14159265358979323846

        phase_rel = rel['real'] / self.range * pi

        real_rel = torch.cos(phase_rel)
        imag_rel = torch.sin(phase_rel)

        re_score = real_rel * tail['real'] + imag_rel * tail['imag']
        im_score = real_rel * tail['imag'] - imag_rel * tail['real']
        re_score = re_score - head['real']
        im_score = im_score - head['imag']

        scores = torch.stack([re_score, im_score], dim=0)

        if summation:
            scores = self.gamma - scores.sum(dim=-1)
        else:
            scores = self.gamma / scores.size(1) - scores

        return scores

    @staticmethod
    def embedding_constraint():
        constraints = {'entity': ['real', 'imag'],
                       'relation': ['real', 'imag']}


@Transformation.register(name="chrono_rotation_tf")
class ChronoRotationTransflormation(Transformation):
    def __init__(self, config):
        super(ChronoRotationTransflormation, self).__init__(config=config)

    def forward(self, head: Dict[str, torch.Tensor], rel: Dict[str, torch.Tensor], tail: Dict[str, torch.Tensor],
                summation=True):
        if not summation:
            raise NotImplementedError

        mat_head = torch.cat(head.values(), dim=2)
        mat_rel = torch.cat(rel.values(), dim=2)
        mat_tail = torch.cat(tail.values(), dim=2)

        rotated_head = [mat_head[:, :, 0] * mat_rel[:, :, 0] - mat_head[:, :, 1] * mat_rel[:, :, 1],
                        -mat_head[:, :, 1] * mat_rel[:, :, 0] - mat_head[:, :, 0] * mat_rel[:, :, 1]]  # complex product
        rotated_head = torch.cat(rotated_head, dim=2)

        ab = torch.matmul(rotated_head, mat_tail.permute((0, 2, 1)))
        ab = torch.einsum('bii->b', ab)

        aa = torch.matmul(rotated_head, rotated_head.permute((0, 2, 1)))
        aa = torch.einsum('bii->b', aa)

        bb = torch.matmul(mat_tail, mat_tail.permute((0, 2, 1)))
        bb = torch.einsum('bii->b', bb)

        scores = ab / torch.sqrt(aa * bb)

        return scores

    @staticmethod
    def embedding_constraint():
        constraints = {'entity': ['real', 'imag'],
                       'relation': ['real', 'imag']}


@Transformation.register(name="rigid_tf")
class RigidTransformation(Transformation):
    pass


@Transformation.register(name="bilinear_tf")
class BilinearTransformation(Transformation):
    pass


@Transformation.register(name="distmult_tf")
class DistMult(Transformation):
    dropout = NumberParam('dropout', default_value=0.4)  # range (0, 1)

    def __init__(self, config):
        super(DistMult, self).__init__(config=config)

    def forward(self, head, rel, tail, summation=True):
        scores = head['real'] * rel['real'] * tail['real']

        if summation:
            scores = scores.sum(dim=-1)

        return scores

    @staticmethod
    def embedding_constraint():
        constraints = {'entity': ['real'],
                       'relation': ['real']}

        return constraints


@Transformation.register(name="complex_factorization_tf")
class ComplexFactorizationTransformation(Transformation):
    def __init__(self, config):
        super(ComplexFactorizationTransformation, self).__init__(config=config)

        self.flatten = True

    def _forward(self, input: Dict):
        self.config.assert_true('head' in input, "Missing head entity")
        self.config.assert_true('rel' in input, "Missing rel entity")
        self.config.assert_true('tail' in input, "Missing tail entity")

    def forward(self, U: Dict, V: Dict, W: Dict, summation=True):
        """
        U, V, W should be Dict[str, torch.Tensor], keys are 'real' and 'imag'
        """
        self.config.assert_true(isinstance(U, dict), "U should be of type dict.")
        self.config.assert_true(isinstance(V, dict), "U should be of type dict.")
        self.config.assert_true(isinstance(W, dict), "U should be of type dict.")
        #
        # assert ['real', 'imag'] in U.keys()
        # assert ['real', 'imag'] in V.keys()
        # assert ['real', 'imag'] in W.keys()
        if self.flatten:
            scores = (U['real'] * V['real'] - U['imag'] * V['imag']) * W['real'] + \
                     (U['imag'] * V['real'] + U['real'] * V['imag']) * W['imag']

            if summation:
                scores = torch.sum(scores, dim=1)

        else:
            raise NotImplementedError
            scores = (U['real'] * V['real'] - U['imag'] * V['imag']) @ W['real'].t() + \
                     (U['imag'] * V['real'] + U['real'] * V['imag']) @ W['imag'].t()

            print(scores.size())

        return scores

    @staticmethod
    def embedding_constraint():
        constraints = {'entity': ['real', 'imag'],
                       'relation': ['real', 'imag']}

        return constraints
