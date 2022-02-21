import torch
from torch import nn
from torch import Tensor

from typing import Type, Callable, Dict
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer
from tkge.data.dataset import DatasetProcessor

from abc import ABC, abstractmethod


class BaseEmbedding(ABC, nn.Module, Configurable):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        nn.Module.__init__(self)
        Configurable.__init__(self, config=config)

        self.dataset = dataset

    # def __init__(self):
    # self._num = num
    # self._dim = dim
    #
    # def dim(self):
    #     return self._num

    # def num(self):
    #     return self._dim

    # def num_embeddings(self):
    #     return self.weight.size(0)
    #
    # def embedding_dim(self):
    #     return self.weight.size(1)

    def initialize(self, type):
        init_dict = {
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_
        }

        return init_dict[type]


class EntityEmbedding(BaseEmbedding):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(EntityEmbedding, self).__init__(config=config, dataset=dataset)
        self._pos_aware = config.get('model.embedding.entity.pos_aware')

        self._head = {}
        self._tail = {}

        self.register_embedding()

    @property
    def head(self):
        return self._head

    @property
    def tail(self):
        if self._pos_aware:
            return self._tail
        else:
            return self._head

    def register_embedding(self):
        for k in self.config.get('model.embedding.entity.keys'):
            embedding_dim = self.config.get(f"model.embedding.entity.keys.{k}.dim") if self.config.get(
                f"model.embedding.global.dim") == -1 else self.config.get(f"model.embedding.global.dim")
            init_method = self.config.get(
                self.config.get(f"model.embedding.entity.keys.{k}.init")) if isinstance(self.config.get(
                f"model.embedding.global.init"), type(None)) else self.config.get(f"model.embedding.global.init")

            self._head[k] = nn.Embedding(num_embeddings=self.dataset.num_entities(),
                                         embedding_dim=embedding_dim)
            self.initialize(init_method)(self._head[k].weight)

            if self._pos_aware:
                self._tail[k] = nn.Embedding(num_embeddings=self.dataset.num_entities(),
                                             embedding_dim=embedding_dim)
                self.initialize(init_method)(self._tail[k].weight)
            else:
                self._tail = self._head

        self._head = nn.ModuleDict(self._head)
        self._tail = nn.ModuleDict(self._tail)

    def __call__(self, index: torch.Tensor, pos: str):
        self.config.assert_true(pos in ['head', 'tail'], f"pos should be either head or tail")

        if pos == 'head':
            return {k: v(index) for k, v in self._head.items()}
        else:
            return {k: v(index) for k, v in self._tail.items()}


class RelationEmbedding(BaseEmbedding):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(RelationEmbedding, self).__init__(config=config, dataset=dataset)

        self._relation = {}
        self._inverse_relation = {}

        self.register_embedding()

    @property
    def relation(self):
        return self._relation

    @property
    def inverse_relation(self):
        return self._relation

    def register_embedding(self):
        num_emb = self.dataset.num_relations() // 2 if self.config.get(
            'task.reciprocal_training') else self.dataset.num_relations()
        num_emb = num_emb * 2 if self.config.get("model.scorer.inverse") or self.config.get(
            "task.reciprocal_training") else num_emb

        for k in self.config.get('model.embedding.relation.keys'):
            embedding_dim = self.config.get(f"model.embedding.relation.keys.{k}.dim") if self.config.get(
                f"model.embedding.global.dim") == -1 else self.config.get(f"model.embedding.global.dim")
            init_method = self.config.get(
                self.config.get(f"model.embedding.relation.keys.{k}.init")) if not self.config.get(
                f"model.embedding.global.init") else self.config.get(f"model.embedding.global.init")

            self._relation[k] = nn.Embedding(num_embeddings=num_emb,
                                             embedding_dim=embedding_dim)

            self.initialize(init_method)(self._relation[k].weight)

        self._relation = nn.ModuleDict(self._relation)
        self._num_emb = num_emb

    def __call__(self, index: torch.Tensor, inverse_relation: bool = False):
        if not inverse_relation:
            return {k: v(index) for k, v in self._relation.items()}
        else:
            if not self.config.get("model.scorer.inverse"):
                raise NotImplementedError('Inverse relations are disabled')
            else:
                inv_index = index - index % 2 + (index + 1) % 2
                return {k: v((inv_index)) for k, v in self._relation.items()}


class TemporalEmbedding(BaseEmbedding):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(TemporalEmbedding, self).__init__(config=config, dataset=dataset)

        self._temporal = {}

        self.register_embedding()

    def register_embedding(self):
        for k in self.config.get('model.embedding.temporal.keys'):
            embedding_dim = self.config.get(f"model.embedding.temporal.keys.{k}.dim") if self.config.get(
                f"model.embedding.global.dim") == -1 else self.config.get(f"model.embedding.global.dim")
            init_method = self.config.get(
                self.config.get(f"model.embedding.temporal.keys.{k}.init")) if not self.config.get(
                f"model.embedding.global.init") else self.config.get(f"model.embedding.global.init")

            self._temporal[k] = nn.Embedding(num_embeddings=self.dataset.num_time_identifier(),
                                             embedding_dim=embedding_dim)
            self.initialize(init_method)(self._temporal[k].weight)

        self._temporal = nn.ModuleDict(self._temporal)

    def get_weight(self, key):
        return self._temporal[key].weight

    def __call__(self, index: torch.Tensor):
        return {k: v(index) for k, v in self._temporal.items()}


class FunctionalTemporalEmbedding(BaseEmbedding):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(FunctionalTemporalEmbedding, self).__init__(config=config, dataset=dataset)

        dim = self.config.get("model.embedding.global.dim")
        init_type = self.config.get("model.embedding.global.init")
        t_min = self.config.get("model.embedding.global.t_min")
        t_max = self.config.get("model.embedding.global.t_max")

        self.freq = nn.Parameter(data=torch.zeros([1, dim // 2]), requires_grad=True)
        torch.nn.init.uniform_(self.freq.data, a=t_min, b=t_max)

    def __call__(self, timestamps: torch.Tensor):
        timestamps = timestamps.squeeze().unsqueeze(-1)
        assert timestamps.dim() == 2 and timestamps.size(1) == 1, f"timestamp {timestamps.size()}"

        omega = 1 / self.freq
        sin_feat = torch.sin(timestamps * omega)
        cos_feat = torch.cos(timestamps * omega)
        feat = torch.cat((sin_feat, cos_feat), dim=1)

        return {'real': feat}


class ExtendedBochnerTemporalEmbedding(BaseEmbedding):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(ExtendedBochnerTemporalEmbedding, self).__init__(config=config, dataset=dataset)

        # dim = self.config.get("model.embedding.global.dim")
        dim = self.config.get("model.embedding.temporal.keys.real.dim")
        init_type = self.config.get("model.embedding.global.init")
        t_min = self.config.get("model.embedding.global.t_min")
        t_max = self.config.get("model.embedding.global.t_max")

        self.freq = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.amps = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.phas = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        torch.nn.init.uniform_(self.freq.data, a=t_min, b=t_max)
        torch.nn.init.xavier_uniform_(self.amps.data)
        torch.nn.init.uniform_(self.phas.data, a=0, b=t_max)

    def __call__(self, timestamps: torch.Tensor):
        timestamps = timestamps.squeeze().unsqueeze(-1)
        assert timestamps.dim() == 2 and timestamps.size(1) == 1, f"timestamp {timestamps.size()}"

        omega = 1 / self.freq
        feat = self.amps * torch.sin(timestamps * omega + self.phas)
        # cos_feat = self.amps * torch.cos(timestamps * omega + self.phas)
        # feat = torch.cat((sin_feat, cos_feat), dim=1)

        return {'real': feat}


class CompositeBochnerTemporalEmbedding(BaseEmbedding):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(CompositeBochnerTemporalEmbedding, self).__init__(config=config, dataset=dataset)

        dim = self.config.get("model.embedding.global.dim")
        init_type = self.config.get("model.embedding.global.init")
        t_min = self.config.get("model.embedding.global.t_min")
        t_max = self.config.get("model.embedding.global.t_max")
        se = self.config.get("model.embedding.global.se")

        se_dim = int(se * dim)
        de_dim = dim - se_dim

        self.se_part = nn.Parameter(data=torch.zeros([1, se_dim]), requires_grad=True)
        self.de_freq = nn.Parameter(data=torch.zeros([1, de_dim]), requires_grad=True)
        self.de_amps = nn.Parameter(data=torch.zeros([1, de_dim]), requires_grad=True)
        self.de_phas = nn.Parameter(data=torch.zeros([1, de_dim]), requires_grad=True)

        torch.nn.init.uniform_(self.se_part.data, a=t_min, b=t_max)
        torch.nn.init.uniform_(self.de_freq.data, a=t_min, b=t_max)
        torch.nn.init.xavier_uniform_(self.de_amps.data)
        torch.nn.init.uniform_(self.de_phas.data, a=0, b=t_max)

    def __call__(self, timestamps: torch.Tensor):
        timestamps = timestamps.squeeze().unsqueeze(-1)
        assert timestamps.dim() == 2 and timestamps.size(1) == 1, f"timestamp {timestamps.size()}"

        omega = 1 / self.de_freq
        feat = self.de_amps * torch.sin(timestamps * omega + self.de_phas)
        bs = timestamps.shape[0]
        feat = torch.cat((self.se_part.repeat((bs, 1)), feat), dim=1)

        return {'real': feat}
# class EntityEmbedding(BaseEmbedding):
#     def __init__(self, num: int, dim: int, pos_aware: bool = False, interleave: bool = False,
#                  expanded_dim_ratio: int = 1):
#         self._num: int = num
#         self._dim: int = dim
#         self._pos_aware: bool = pos_aware
#         self._interleave: bool = interleave
#
#         expanded_num_ratio = 2 if self._pos_aware else 1
#
#         super(EntityEmbedding, self).__init__(num * expanded_num_ratio, dim * expanded_dim_ratio)
#
#     def dim(self):
#         return self._dim
#
#     def num(self):
#         return self._num
#
#     def get_by_index(self, index: torch.Tensor):
#         """
#         behave same as __index__ when pos_aware is false and return head and tail embeddings of the entities
#         """
#         pass


# class RelationEmbedding(BaseEmbedding):
#     def __init__(self, num: int, dim: int, reciprocal: bool = False, interleave: bool = False,
#                  expanded_dim_ratio: int = 1):
#         self._num: int = num
#         self._dim: int = dim
#         self._reciprocal: bool = reciprocal
#         self._interleave: bool = interleave
#
#         expanded_num_ratio = 2 if self._reciprocal else 1
#
#         super(RelationEmbedding, self).__init__(num * expanded_num_ratio, dim * expanded_dim_ratio)
#
#     def dim(self):
#         return self._dim
#
#     def num(self):
#         return self._num
#
#     def get_by_index(self, index: torch.Tensor):
#         """
#         behave same as __index__ when pos_aware is false and return original and reciprocal embeddings of the relations
#         """
#         pass


# class RealEntityEmbedding(EntityEmbedding):
#     def __init__(self, num: int, dim: int, pos_aware: bool = False, interleave: bool = False):
#         super(RealEntityEmbedding, self).__init__(num=num, dim=dim, pos_aware=pos_aware, interleave=interleave,
#                                                   expanded_dim_ratio=1)
#
#
# class ComplexEntityEmbedding(EntityEmbedding):
#     def __init__(self, num: int, dim: int, pos_aware: bool = False, interleave: bool = False):
#         super(ComplexEntityEmbedding, self).__init__(num=num, dim=dim, pos_aware=pos_aware, interleave=interleave,
#                                                      expanded_dim_ratio=2)
#
#     def re(self):
#         pass
#
#     def im(self):
#         pass
#
#     class TranslationRelationEmbedding(RelationEmbedding):
#         pass
#
#     class QuaternionRelationEmbedding(RelationEmbedding):
#         pass
#
#     class DualQuaternionRelationEmbedding(RelationEmbedding):
#         pass
#
#     # class BaseEmbedder(nn.Embedding):
#     #     """
#     #     Base class for all embedders of a fixed number of objects including entities, relations and timestamps
#     #     """
#     #
#     #     def __init__(self,
#     #                  num_embeddings: int,
#     #                  embedding_dim: int,
#     #                  initializer: str = "uniform",
#     #                  init_args: Dict = {"from_": 0, "to": 0},
#     #                  reg: str = "renorm"):
#     #         super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
#     #
#     #         if initializer == "uniform":
#     #             self.weight.data.uniform_(**init_args)
#     #         else:
#     #             raise NotImplementedError
#     #
#     #         self.regularizer = Regularizer.create(reg)  # TODO: modify regularizer
#     #
#     #     def forward(self, indexes: Tensor) -> Tensor:
#     #         return self.embed(indexes)
#     #
#     #     def embed(self, indexes: Tensor) -> Tensor:
#     #         raise NotImplementedError
#     #
#     #     def embed_all(self) -> Tensor:
#     #         raise NotImplementedError
#     #
#     #     def regularize(self):  # TODO should it be inplace?
#     #
#     #         pass
#     #
#     #     def split(self, sep):
#     #         return self.embed(Tensor[range(sep)])
#     #
#     #     def __getitem__(self, item):
#     #         pass
#
#     # class BaseEmbedding(nn.Module, Registrable):
#     #     def __init__(self):
#     #         super(BaseEmbedding, self).__init__()
#     #
#     #         self.params: Dict[str, Tensor] = defaultdict(dict)
#     #
#     #     def build(self):
#     #         raise NotImplementedError
#     #
#     #         ## build up all embeddings needed at one time, and do the regularization
#     #         # self.params["new_key"] = nn.Parameter(num_emb, emb_dim)
#     #         # self.params["new_key"].init()
#     #
#     #     def forward(self, indexes):
#     #         raise NotImplementedError
#     #
#     #         ## return the score of indexed embedding
#     #         # return embedding_package
#     #
#     # class TkgeModel(nn.Module, Registrable):
#     #     def __init__(self, config: Config):
#     #         nn.Module().__init__()
#     #         Registrable().__init__(config=config)
#     #
#     #         # Customize
#     #         self._relation_embedder = BaseEmbedder()
#     #         self._entity_embedder = BaseEmbedder()
#     #
#     #     def forward(self):
#     #         # 模型主要代码写在这里
#     #         # return score
#     #         pass
#     #
#     #     def load_chpt(self):
#     #         pass
