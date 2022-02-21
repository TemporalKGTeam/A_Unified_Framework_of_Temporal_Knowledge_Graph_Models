import torch
from torch import nn
from torch import Tensor

from typing import Type, Callable, Dict, Union, Optional
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer
from tkge.models.embedding import *


class Embedder():
    pass


class LookUpEmbedder():
    """
    input: index
    output: embedding
    """
    pass


class EmbeddingSpace(nn.Module, Registrable, Configurable):
    r"""EmbeddingSpace retrieves embeddings by index from a bundle of heterogeneous embeddings.

    Args:
        num_embeddings (int)
        embedding_dim (int)
        max_norm (float)
        norm_type:

    Attributes:
        entity:
        relation:
        temporal:
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

        self._entity: EntityEmbedding = None
        self._relation: RelationEmbedding = None
        self._temporal: Optional[Union[TemporalEmbedding, torch.nn.ModuleDict[TemporalEmbedding]]] = None

    @classmethod
    def from_pretrained(cls) -> 'EmbeddingSpace':
        r"""Creates EmbeddingSpace instance from pretrained checkpoint.

        Examples::


        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Config) -> 'EmbeddingSpace':
        r"""Creates new EmbeddingSpace configured by configuration file

        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, **params) -> 'EmbeddingSpace':
        r"""Creates new EmbeddingSpace defined by params

        """
        raise NotImplementedError

    @property
    def entity(self):
        return self._entity

    @property
    def relation(self):
        return self._relation

    @property
    def temporal(self):
        return self._temporal

    def get_ent_emb(self, index):
        return self._entity.get_by_index(index)

    def get_rel_emb(self, index):
        return self._relation.get_by_index(index)

    def get_temp_emb(self, index):
        if isinstance(self._temporal, type(None)):
            return None

        elif isinstance(self._temporal, torch.nn.ModuleDict):
            return {k: v.get_by_index(index) for k, v in self._temporal.item()}

        elif isinstance(self._temporal, TemporalEmbedding):
            return self._temporal.get_by_index(index)

        else:
            raise NotImplementedError

    def forward(self, index_inputs: torch.Tensor):
        """
        args:
            index_inputs (torch.LongTensor): organized as SPOT
        """
        return {'s': self.get_ent_emb(index_inputs, 'head'), 'p': self.get_rel_emb(index_inputs),
                'o': self.get_ent_emb(index_inputs, 'tail'), 't': self.get_temp_emb(index_inputs)}


@EmbeddingSpace.register(name='static_embedding_space')
class StaticEmbeddingSpace(EmbeddingSpace):
    def __init__(self):
        super(StaticEmbeddingSpace, self).__init__()


@EmbeddingSpace.register(name='temporal_embedding_space')
class TemporalEmbeddingSpace(EmbeddingSpace):
    def __init__(self):
        super(TemporalEmbeddingSpace, self).__init__()


@EmbeddingSpace.register(name='diachronic_entity_embedding_space')
class DiachronicEntityEmbeddingSpace(EmbeddingSpace):
    def __init__(self):
        super(DiachronicEntityEmbeddingSpace, self).__init__()
