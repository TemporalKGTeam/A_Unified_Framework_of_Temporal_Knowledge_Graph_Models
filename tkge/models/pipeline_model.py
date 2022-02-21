import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from enum import Enum
import os
from collections import defaultdict
from typing import Mapping, Dict, List, Any
import random

from tkge.common.registrable import Registrable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.data.dataset import DatasetProcessor
from tkge.models.layers import LSTMModel
from tkge.models.model import BaseModel
from tkge.models.utils import *
from tkge.models.embedding_space import *
from tkge.models.fusion import TemporalFusion
from tkge.models.transformation import Transformation


@BaseModel.register(name='pipeline_model')
class PipelineModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(PipelineModel, self).__init__(config=config, dataset=dataset)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
            self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

        print(self._entity_embeddings.state_dict())

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        # if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
        #     if samples.size(1)==4:
        #         temp_index = samples[:, -1]
        #         temp.update(self._temporal_embeddings(temp_index.long()))
        #     else:
        #         temp_indexes = samples[:, 3:]
        #         for i in range(temp_indexes.size(1)):
        #             temp_embs = self._temporal_embeddings(temp_indexes[:, i:i + 1].long())
        #             temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
        #             temp.update(temp_embs)

        # if self.config.get('dataset.temporal.float'):
        #     temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
        #     for i in range(temp_float.size(1)):
        #         temp.update({f"level{i}": temp_float[:, i:i + 1]})

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        # fusion

        # get encoded embeddings
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'],
                                              summation=False)

            scores = (scores + scores_inv) / 2

        scores = F.dropout(scores, p=self.config.get("model.fusion.p"), training=self.training)
        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores, p=self.config.get(
                'model.transformation.p'), dim=1)
        else:
            scores = torch.sum(scores, dim=1)

        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - scores

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False

        # TODO 1vsAll or negative sampling
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True

        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='translation_simple_model')
class TransSimpleModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(TransSimpleModel, self).__init__(config=config, dataset=dataset)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        if self.config.get('dataset.temporal.index'):
            self._temporal_embeddings = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

        self.dropout = torch.nn.Dropout(p=self.config.get('model.p'))

        # import pprint
        #
        # pprint.pprint({n: p.size() for n, p in self.named_parameters()})
        # assert False

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        # fusion

        # get encoded embeddings
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        # dropot
        # fused_spo_emb['s']['real'] = self.dropout(fused_spo_emb['s']['real'])
        # fused_spo_emb['p']['real'] = self.dropout(fused_spo_emb['p']['real'])
        # fused_spo_emb['o']['real'] = self.dropout(fused_spo_emb['o']['real'])
        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self._inverse_scorer:
            # fused_spo_emb_inv['s']['real'] = self.dropout(fused_spo_emb_inv['s']['real'])
            # fused_spo_emb_inv['p']['real'] = self.dropout(fused_spo_emb_inv['p']['real'])
            # fused_spo_emb_inv['o']['real'] = self.dropout(fused_spo_emb_inv['o']['real'])
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'],
                                              summation=False)

            scores = (scores + scores_inv) / 2

        scores = F.dropout(scores, p=self.config.get("model.fusion.p"), training=self.training)
        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores, p=self.config.get(
                'model.transformation.p'), dim=1)
        else:
            scores = torch.sum(scores, dim=1)

        factors = {
            "n3": (torch.sqrt(self._entity_embeddings._head['real'].weight ** 2),
                   torch.sqrt(self._entity_embeddings._tail['real'].weight ** 2),
                   torch.sqrt(self._relation_embeddings._relation['real'].weight ** 2)),
            "lambda3": self._temporal_embeddings.get_weight('real')
        }

        return scores, factors

    def _fuse(self, spot_emb: Dict[str, Any], fuse_target: str):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True

        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor


@BaseModel.register(name='translation_transe_model')
class TransSimpleModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(TransSimpleModel, self).__init__(config=config, dataset=dataset)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        if self.config.get('dataset.temporal.index'):
            self._temporal_embeddings = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

        # import pprint
        #
        # pprint.pprint({n: p.size() for n, p in self.named_parameters()})
        # assert False

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        # fusion

        # get encoded embeddings
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        # dropot
        # fused_spo_emb['s']['real'] = self.dropout(fused_spo_emb['s']['real'])
        # fused_spo_emb['p']['real'] = self.dropout(fused_spo_emb['p']['real'])
        # fused_spo_emb['o']['real'] = self.dropout(fused_spo_emb['o']['real'])

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self.training:
            mask = torch.rand_like(scores) < (1 - self.config.get("model.fusion.p"))
        else:
            mask = torch.ones_like(scores)

        scores = self.config.get('model.transformation.gamma') - torch.norm(scores * mask, p=self.config.get(
            'model.transformation.p'), dim=1)

        if self._inverse_scorer:
            # fused_spo_emb_inv['s']['real'] = self.dropout(fused_spo_emb_inv['s']['real'])
            # fused_spo_emb_inv['p']['real'] = self.dropout(fused_spo_emb_inv['p']['real'])
            # fused_spo_emb_inv['o']['real'] = self.dropout(fused_spo_emb_inv['o']['real'])
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'],
                                              summation=False)

            scores_inv = self.config.get('model.transformation.gamma') - torch.norm(scores_inv * mask,
                                                                                    p=self.config.get(
                                                                                        'model.transformation.p'),
                                                                                    dim=1)

            scores = (scores + scores_inv) / 2

        # scores = F.dropout(scores, p=self.config.get("model.fusion.p"), training=self.training)
        # if self.config.get('model.transformation.type') == 'translation_tf':
        #     scores = self.config.get('model.transformation.gamma') - torch.norm(scores, p=self.config.get(
        #         'model.transformation.p'), dim=1)
        # else:
        #     scores = torch.sum(scores, dim=1)

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb: Dict[str, Any], fuse_target: str):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True

        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor


@BaseModel.register(name='static_translation_simple_model')
class StaticTransSimpleModel(BaseModel):
    def encoding(self, indexes):
        new_indexes = torch.zeros_like(indexes).long()

        return {'real': self._temporal_embedder(new_indexes)}

    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(StaticTransSimpleModel, self).__init__(config=config, dataset=dataset)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        self._temporal_embedder = nn.Embedding(num_embeddings=1,
                                               embedding_dim=self.config.get("model.embedding.global.dim"))

        if self.config.get('dataset.temporal.index'):
            self._temporal_embeddings = self.encoding

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

        self.dropout = torch.nn.Dropout(p=self.config.get('model.p'))

        # import pprint
        #
        # pprint.pprint({n: p.size() for n, p in self.named_parameters()})
        # assert False

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        # fusion

        # get encoded embeddings
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        # dropot
        # fused_spo_emb['s']['real'] = self.dropout(fused_spo_emb['s']['real'])
        # fused_spo_emb['p']['real'] = self.dropout(fused_spo_emb['p']['real'])
        # fused_spo_emb['o']['real'] = self.dropout(fused_spo_emb['o']['real'])
        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            # fused_spo_emb_inv['s']['real'] = self.dropout(fused_spo_emb_inv['s']['real'])
            # fused_spo_emb_inv['p']['real'] = self.dropout(fused_spo_emb_inv['p']['real'])
            # fused_spo_emb_inv['o']['real'] = self.dropout(fused_spo_emb_inv['o']['real'])
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {
            "n3": (torch.sqrt(self._entity_embeddings._head['real'].weight ** 2),
                   torch.sqrt(self._entity_embeddings._tail['real'].weight ** 2),
                   torch.sqrt(self._relation_embeddings._relation['real'].weight ** 2)),
            # "lambda3": self._temporal_embeddings.get_weight('real')
        }

        return scores, factors

    def _fuse(self, spot_emb: Dict[str, Any], fuse_target: str):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor


@BaseModel.register(name='de_pipeline_model')
class DePipelineModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(DePipelineModel, self).__init__(config=config, dataset=dataset)

        se = self.config.get("model.fusion.se")
        self.config.set("model.embedding.entity.keys.ent_embs.dim",
                        round(self.config.get("model.embedding.global.dim") * se))
        self.config.set("model.embedding.entity.keys.amps_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.amps_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.amps_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.relation.keys.real.dim",
                        round(self.config.get("model.embedding.global.dim")))

        self.config.set("model.embedding.global.dim", -1)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
            self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

        # import pprint
        #
        # pprint.pprint({n: p.size() for n, p in self.named_parameters()})
        # assert False

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='de_pipeline_dropout_model')
class DePipelineDropoutModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(DePipelineDropoutModel, self).__init__(config=config, dataset=dataset)

        se = self.config.get("model.fusion.se")
        self.config.set("model.embedding.entity.keys.ent_embs.dim",
                        round(self.config.get("model.embedding.global.dim") * se))
        self.config.set("model.embedding.entity.keys.amps_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.amps_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.amps_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.relation.keys.real.dim",
                        round(self.config.get("model.embedding.global.dim")))

        self.config.set("model.embedding.global.dim", -1)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
            self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self.training:
            mask = torch.rand_like(scores) < (1 - self.config.get("model.fusion.p"))
        else:
            mask = torch.ones_like(scores)

        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores * mask, p=self.config.get(
                'model.transformation.p'), dim=1)

            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores_inv = self.config.get('model.transformation.gamma') - torch.norm(scores_inv * mask,
                                                                                        p=self.config.get(
                                                                                            'model.transformation.p'),
                                                                                        dim=1)

                scores = (scores + scores_inv) / 2
        else:
            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores = (scores + scores_inv) * mask / 2

            scores = torch.sum(scores, dim=1)

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='utee_pipeline_dropout_model')
class UteePipelineDropoutModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(UteePipelineDropoutModel, self).__init__(config=config, dataset=dataset)

        se = self.config.get("model.fusion.se")
        self.config.set("model.embedding.entity.keys.real.dim",
                        round(self.config.get("model.embedding.global.dim") * se))
        self.config.set("model.embedding.relation.keys.real.dim",
                        round(self.config.get("model.embedding.global.dim")))
        self.config.set("model.embedding.temporal.keys.real.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))

        # de_dim = round(self.config.get("model.embedding.global.dim") * (1 - se))

        # self.config.set("model.embedding.entity.keys.amps_y.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.amps_m.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.amps_d.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.freq_y.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.freq_m.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.freq_d.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.phi_y.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.phi_m.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.entity.keys.phi_d.dim",
        #                 round(self.config.get("model.embedding.global.dim") * (1 - se)))
        # self.config.set("model.embedding.relation.keys.real.dim",
        #                 round(self.config.get("model.embedding.global.dim")))

        self.config.set("model.embedding.global.dim", -1)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        self._temporal_embeddings = ExtendedBochnerTemporalEmbedding(config=config, dataset=dataset)

        # temp_emb_dict = {
        #     'amps_y': nn.Embedding(1, de_dim),
        #     'amps_m': nn.Embedding(1, de_dim),
        #     'amps_d': nn.Embedding(1, de_dim),
        #     'freq_y': nn.Embedding(1, de_dim),
        #     'freq_m': nn.Embedding(1, de_dim),
        #     'freq_d': nn.Embedding(1, de_dim),
        #     'phi_y': nn.Embedding(1, de_dim),
        #     'phi_m': nn.Embedding(1, de_dim),
        #     'phi_d': nn.Embedding(1, de_dim),
        # }

        # self.temp_emb_dict = nn.ModuleDict(temp_emb_dict)
        #
        # if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
        #     self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        # temp_emb = {k: v(torch.tensor(0).long().to(head)) for k, v in self.temp_emb_dict.items()}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self.training:
            mask = torch.rand_like(scores) < (1 - self.config.get("model.fusion.p"))
        else:
            mask = torch.ones_like(scores)

        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores * mask, p=self.config.get(
                'model.transformation.p'), dim=1)

            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores_inv = self.config.get('model.transformation.gamma') - torch.norm(scores_inv * mask,
                                                                                        p=self.config.get(
                                                                                            'model.transformation.p'),
                                                                                        dim=1)

                scores = (scores + scores_inv) / 2
        else:
            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores = (scores + scores_inv) * mask / 2

            scores = torch.sum(scores, dim=1)

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        # if hasattr(self, '_temporal_embeddings'):
        #     factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='de_pipeline_dropout_reg_model')
class DePipelineDropoutRegModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(DePipelineDropoutRegModel, self).__init__(config=config, dataset=dataset)

        se = self.config.get("model.fusion.se")
        self.config.set("model.embedding.entity.keys.ent_embs.dim",
                        round(self.config.get("model.embedding.global.dim") * se))
        self.config.set("model.embedding.entity.keys.amps_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.amps_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.amps_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.freq_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_y.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_m.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.entity.keys.phi_d.dim",
                        round(self.config.get("model.embedding.global.dim") * (1 - se)))
        self.config.set("model.embedding.relation.keys.real.dim",
                        round(self.config.get("model.embedding.global.dim")))

        self.config.set("model.embedding.global.dim", -1)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
            self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self.training:
            mask = torch.rand_like(scores) < (1 - self.config.get("model.fusion.p"))
        else:
            mask = torch.ones_like(scores)

        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores * mask, p=self.config.get(
                'model.transformation.p'), dim=1)

            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores_inv = self.config.get('model.transformation.gamma') - torch.norm(scores_inv * mask,
                                                                                        p=self.config.get(
                                                                                            'model.transformation.p'),
                                                                                        dim=1)

                scores = (scores + scores_inv) / 2
        else:
            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores = (scores + scores_inv) * mask / 2

            scores = torch.sum(scores, dim=1)

        _static_ent = [v for k, v in self._entity_embeddings.named_parameters() if 'ent_embs' in k]
        # _dynamic_ent = [v for k, v in self._entity_embeddings.named_parameters() if 'ent_embs' not in k]
        _freq_ent = [v for k, v in self._entity_embeddings.named_parameters() if 'freq' in k]

        factors = {"static_entity_reg": _static_ent,
                   # "dynamic_entity_reg": _dynamic_ent,
                   "dynamic_entity_reg": _freq_ent,
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='atise_pipeline_model')
class ATiSEPipelineModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(ATiSEPipelineModel, self).__init__(config=config, dataset=dataset)

        self.config.set("model.embedding.entity.keys.emb.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.emb_T.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.alpha.dim",
                        1)
        self.config.set("model.embedding.entity.keys.beta.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.omega.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.var.dim",
                        self.config.get("model.embedding.global.dim"))

        self.config.set("model.embedding.relation.keys.emb.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.emb_T.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.alpha.dim",
                        1)
        self.config.set("model.embedding.relation.keys.beta.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.omega.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.var.dim",
                        self.config.get("model.embedding.global.dim"))

        self.config.set("model.embedding.global.dim", -1)

        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
            self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='atise_pipeline_dropout_model')
class ATiSEPipelineDropoutModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(ATiSEPipelineDropoutModel, self).__init__(config=config, dataset=dataset)

        self.config.set("model.embedding.entity.keys.emb.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.emb_T.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.alpha.dim",
                        1)
        self.config.set("model.embedding.entity.keys.beta.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.omega.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.entity.keys.var.dim",
                        self.config.get("model.embedding.global.dim"))

        self.config.set("model.embedding.relation.keys.emb.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.emb_T.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.alpha.dim",
                        1)
        self.config.set("model.embedding.relation.keys.beta.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.omega.dim",
                        self.config.get("model.embedding.global.dim"))
        self.config.set("model.embedding.relation.keys.var.dim",
                        self.config.get("model.embedding.global.dim"))

        self.config.set("model.embedding.global.dim", -1)

        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
            self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self.training:
            mask = torch.rand_like(scores) < (1 - self.config.get("model.fusion.p"))
        else:
            mask = torch.ones_like(scores)

        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores * mask, p=self.config.get(
                'model.transformation.p'), dim=1)

            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores_inv = self.config.get('model.transformation.gamma') - torch.norm(scores_inv * mask,
                                                                                        p=self.config.get(
                                                                                            'model.transformation.p'),
                                                                                        dim=1)

                scores = (scores + scores_inv) / 2
        else:
            if self._inverse_scorer:
                scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'],
                                                  fused_spo_emb_inv['o'], summation=False)

                scores = (scores + scores_inv) * mask / 2

            scores = torch.sum(scores, dim=1)

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='bochner_pipeline_model')
class BochnerPipelineModel(BaseModel):
    def functional_encoding(self, timestamps):
        """
        timestamp: shape [num, 1]
        """
        timestamps = timestamps.squeeze().unsqueeze(-1)
        assert timestamps.dim() == 2 and timestamps.size(1) == 1, f"timestamp {timestamps.size()}"

        sin_feat = torch.sin(timestamps * self.sampled_freq)
        cos_feat = torch.cos(timestamps * self.sampled_freq)
        feat = torch.cat((sin_feat, cos_feat), dim=1)

        return {'real': feat}

    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(BochnerPipelineModel, self).__init__(config=config, dataset=dataset)

        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        t_min = self.config.get("model.embedding.global.t_min")
        t_max = self.config.get("model.embedding.global.t_max")
        dim = self.config.get("model.embedding.global.dim")

        sampled_freq = torch.distributions.uniform.Uniform(t_min, t_max).sample([dim // 2])
        sampled_freq = 1. / sampled_freq
        sampled_freq, _ = torch.sort(sampled_freq)
        sampled_freq = sampled_freq.unsqueeze(0)
        self.sampled_freq = sampled_freq.to(self.config.get("task.device"))

        self._temporal_embeddings = self.functional_encoding

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='basis_bochner_pipeline_model')
class BasisBochnerPipelineModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(BasisBochnerPipelineModel, self).__init__(config=config, dataset=dataset)

        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        self._temporal_embeddings = FunctionalTemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='basis_bochner_pipeline_dropout_model')
class BasisBochnerPipelineDropoutModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(BasisBochnerPipelineDropoutModel, self).__init__(config=config, dataset=dataset)

        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        self._temporal_embeddings = FunctionalTemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'],
                                              summation=False)

            scores = (scores + scores_inv) / 2

        scores = F.dropout(scores, p=self.config.get("model.fusion.p"), training=self.training)
        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores, p=self.config.get(
                'model.transformation.p'), dim=1)
        else:
            scores = torch.sum(scores, dim=1)

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='extended_bochner_pipeline_model')
class ExtendedBochnerPipelineModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(ExtendedBochnerPipelineModel, self).__init__(config=config, dataset=dataset)

        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        self._temporal_embeddings = ExtendedBochnerTemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='extended_bochner_pipeline_dropout_model')
class ExtendedBochnerPipelineDropoutModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(ExtendedBochnerPipelineDropoutModel, self).__init__(config=config, dataset=dataset)

        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        self._temporal_embeddings = ExtendedBochnerTemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'],
                                              summation=False)

            scores = (scores + scores_inv) / 2

        scores = F.dropout(scores, p=self.config.get("model.fusion.p"), training=self.training)
        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores, p=self.config.get(
                'model.transformation.p'), dim=1)
        else:
            scores = torch.sum(scores, dim=1)

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='composite_bochner_pipeline_dropout_model')
class CompositeBochnerPipelineDropoutModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(CompositeBochnerPipelineDropoutModel, self).__init__(config=config, dataset=dataset)

        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        self._temporal_embeddings = CompositeBochnerTemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

    @forward_checking
    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'], summation=False)

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'],
                                              summation=False)

            scores = (scores + scores_inv) / 2

        scores = F.dropout(scores, p=self.config.get("model.fusion.p"), training=self.training)
        if self.config.get('model.transformation.type') == 'translation_tf':
            scores = self.config.get('model.transformation.gamma') - torch.norm(scores, p=self.config.get(
                'model.transformation.p'), dim=1)
        else:
            scores = torch.sum(scores, dim=1)

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        self.training = False
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        self.training = True
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


if __name__ == '__main__':
    print(BaseModel.list_available())
