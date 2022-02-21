import torch
from torch import nn

from typing import Type, Callable, Dict
from collections import defaultdict
from functools import reduce
from abc import ABC, abstractmethod

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.models.layers import LSTMModel
from tkge.train.regularization import Regularizer


class TemporalFusion(ABC, nn.Module, Registrable, Configurable):
    def __init__(self, config: Config):
        nn.Module.__init__(self)
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

    @classmethod
    def create_from_name(cls, config: Config):
        fusion_type = config.get("model.fusion.type")
        kwargs = config.get("model.fusion.args")

        kwargs = kwargs if not isinstance(kwargs, type(None)) else {}

        if fusion_type in TemporalFusion.list_available():
            # kwargs = config.get("model.args")  # TODO: get all args params
            return TemporalFusion.by_name(fusion_type)(config, **kwargs)
        else:
            raise ConfigurationError(
                f"{fusion_type} specified in configuration file is not supported"
                f"implement your model class with `TemporalFusion.register(name)"
            )

    @abstractmethod
    def forward(self, operand1, operand2):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def embedding_constraint():
        raise NotImplementedError


@TemporalFusion.register(name="concatenate_fusion")
class ConcatenateTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ConcatenateTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        res = {'real': torch.cat([operand1['real'], operand2['real']], dim=1)}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="addition_fusion")
class AdditionTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(AdditionTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        res = {'real': operand1['real'] + operand2['real']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="elementwise_product_fusion")
class ElementwiseTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ElementwiseTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        res = {'real': operand1['real'] * operand2['real']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="complex_elementwise_product_fusion")
class ComplexElementwiseTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ComplexElementwiseTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        """
        operand1 and operand2 should be complex embeddings
        """
        p = operand1['real'] * operand2['real'], \
            operand1['imag'] * operand2['real'], \
            operand1['real'] * operand2['imag'], \
            operand1['imag'] * operand2['imag']

        res = {'real': p[0] - p[3],
               'imag': p[1] + p[2]}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real', 'imag'],
                          'operand2': ['real', 'imag']}

        out_constraints = {'result': ['real', 'imag']}

        return in_constraints, out_constraints

@TemporalFusion.register(name="complex_addition_fusion")
class ComplexAdditionTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ComplexAdditionTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        """
        operand1 and operand2 should be complex embeddings
        """
        res = {'real': operand1['real'] + operand1['real'],
               'imag': operand1['imag'] + operand1['imag']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real', 'imag'],
                          'operand2': ['real', 'imag']}

        out_constraints = {'result': ['real', 'imag']}

        return in_constraints, out_constraints

@TemporalFusion.register(name="tnt_fusion")
class TNTTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(TNTTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        p = operand1['real'] * operand2['real'], \
            operand1['imag'] * operand2['real'], \
            operand1['real'] * operand2['imag'], \
            operand1['imag'] * operand2['imag']

        res = {'real': p[0] - p[3] + operand2['static_real'],
               'imag': p[1] + p[2] + operand2['static_imag']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real', 'imag'],
                          'operand2': ['real', 'imag', 'static_real', 'static_imag']}

        out_constraints = {'result': ['real', 'imag']}

        return in_constraints, out_constraints



@TemporalFusion.register(name="reproject_fusion")
class ReprojectTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ReprojectTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        """
        input should be [static embedding, temporal embedding]
        """
        inner = torch.sum(operand1['real'] * operand2['real'], dim=1, keepdim=True) / torch.sum(operand1['real'] ** 2,
                                                                                                dim=-1, keepdim=True)
        res = {'real': operand1['real'] - inner * operand2['real']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="hidden_representaion_fusion")
class HiddenRepresentationCombination(TemporalFusion):
    """Base combination operator for hidden representation"""

    def __init__(self, config: Config):
        super(HiddenRepresentationCombination, self).__init__(config=config)

    def forward(self, operand1, operand2):
        raise NotImplementedError


@TemporalFusion.register(name="diachronic_entity_fusion")
class DiachronicEntityFusion(HiddenRepresentationCombination):
    def __init__(self, config: Config):
        super(DiachronicEntityFusion, self).__init__(config=config)

        self.time_nl = torch.sin

    def forward(self, operand1, operand2):
        """
        operand1 are entity embedding
        operand2 are timestamp index

        return a batch_size * dim embedding
        """

        time_emb = operand1['amps_y'] * self.time_nl(
            operand1['freq_y'] * operand2['level0'] + operand1['phi_y'])
        time_emb += operand1['amps_m'] * self.time_nl(
            operand1['freq_m'] * operand2['level1'] + operand1['phi_m'])
        time_emb += operand1['amps_d'] * self.time_nl(
            operand1['freq_d'] * operand2['level2'] + operand1['phi_d'])

        emb = torch.cat((operand1['ent_embs'], time_emb), 1)
        res = {'real': emb}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {
            'operand1': ['ent_embs', 'amps_y', 'amps_m', 'amps_d', 'freq_y', 'freq_m', 'freq_d', 'phi_y', 'phi_m',
                         'phi_d'],
            'operand2': ['level0', 'level1', 'level2']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="time_aware_fusion")
class TimeAwareFusion(HiddenRepresentationCombination):
    def __init__(self, config: Config):
        super(TimeAwareFusion, self).__init__(config=config)

        self.emb_dim = self.config.get("model.embedding.global.dim")
        self.l1_flag = True #self.config.get("model.fusion.l1_flag")
        self.p = self.config.get("model.fusion.p")

        self.dropout = torch.nn.Dropout(p=self.p)
        self.lstm = LSTMModel(self.emb_dim, n_layer=1)

    def forward(self, operand1, operand2):
        token_e = [operand2['level' + str(i) + '_real'] for i in
                   range(len(operand2))]  # level0_real, ..., level7_real, bs*dim
        token_e = reduce(lambda a, b: torch.cat((a, b), dim=1), token_e)
        r_e = operand1['real'].unsqueeze(0).permute(1, 0, 2)
        seq_e = torch.cat((r_e, token_e), dim=1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return {'real': rseq_e}

    @staticmethod
    def embedding_constraint():
        in_constraints = {
            'operand1': ['real'],
            'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints



@TemporalFusion.register(name="atise_fusion")
class ATiSEFusion(HiddenRepresentationCombination):
    def __init__(self, config: Config):
        super(ATiSEFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        pi = 3.14159265358979323846

        mean = operand1['emb'] + operand2['level0'] * operand1['alpha'] * operand1['emb_T'] + operand1[
            'beta'] * torch.sin(2 * pi * operand1['omega'] * operand2['level0'])
        var = operand1['var']

        res = {'real': mean, 'var': var}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {
            'operand1': ['emb', 'emb_T', 'alpha', 'beta', 'omega', 'var'],
            'operand2': ['level0']}

        out_constraints = {'result': ['mean', 'var']}

        return in_constraints, out_constraints
