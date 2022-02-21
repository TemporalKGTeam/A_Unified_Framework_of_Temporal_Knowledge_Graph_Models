import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import math

from collections import defaultdict
from typing import Dict
from abc import ABC, abstractmethod

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.data.dataset import DatasetProcessor
from tkge.models.layers import LSTMModel
from tkge.models.utils import *


class BaseModel(ABC, nn.Module, Registrable, Configurable):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        nn.Module.__init__(self)
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

        self.dataset = dataset

    @staticmethod
    def create(config: Config, dataset: DatasetProcessor):
        model_type = config.get("model.type")
        kwargs = config.get("model.args")

        kwargs = kwargs if not isinstance(kwargs, type(None)) else {}

        if model_type in BaseModel.list_available():
            # kwargs = config.get("model.args")  # TODO: get all args params
            return BaseModel.by_name(model_type)(config, dataset, **kwargs)
        else:
            raise ConfigurationError(
                f"{model_type} specified in configuration file is not supported"
                f"implement your model class with `BaseModel.register(name)"
            )

    def load_config(self):
        # TODO(gengyuan): should undefined params initialized by default values?
        raise NotImplementedError

    def prepare_embedding(self):
        raise NotImplementedError

    def get_embedding(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, samples, **kwargs):
        """
        return scores, factors
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, queries: torch.Tensor):
        """
        Should be a wrapper of method forward or a computation flow same as that in forward.
        Particularly for prediction task with incomplete queries as inputs.
        New modules or learnable parameter constructed in this namespace should be avoided since it's not evolved in training procedure.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, samples: torch.Tensor):
        # TODO(gengyuan): wrapping all the models
        """
        Should be a wrapper of forward or a computation flow same as that in forward.
        This method is intended to handle arbitrarily-shaped samples due to negative sampling, either matrix or flatteded.
        Especially when training procedure and prediction procedure are different.
        Samples should be processed in this method and then passed to forward.
        Input samples are the direct output of the negative sampling.
        """
        raise NotImplementedError


@BaseModel.register(name='de-simple')
class DeSimplEModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        self.prepare_embedding()

        self.time_nl = torch.sin  # TODO add to configuration file

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()

        emb_dim = self.config.get("model.embedding.emb_dim")
        se_prop = self.config.get("model.embedding.se_prop")
        s_emb_dim = int(se_prop * emb_dim)
        t_emb_dim = emb_dim - s_emb_dim

        # torch.manual_seed(0)
        # torch.cuda.manual_seed_all(0)
        # np.random.seed(0)
        # random.seed(0)
        # torch.backends.cudnn.deterministic = True
        # os.environ['PYTHONHASHSEED'] = str(0)

        self.embedding: Dict[str, nn.Module] = defaultdict(dict)

        self.embedding.update({'ent_embs_h': nn.Embedding(num_ent, s_emb_dim)})
        self.embedding.update({'ent_embs_t': nn.Embedding(num_ent, s_emb_dim)})
        self.embedding.update({'rel_embs_f': nn.Embedding(num_rel, s_emb_dim + t_emb_dim)})
        self.embedding.update({'rel_embs_i': nn.Embedding(num_rel, s_emb_dim + t_emb_dim)})

        # frequency embeddings for the entities

        self.embedding.update({'m_freq_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'m_freq_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_freq_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_freq_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_freq_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_freq_t': nn.Embedding(num_ent, t_emb_dim)})

        # phi embeddings for the entities
        self.embedding.update({'m_phi_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'m_phi_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_phi_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_phi_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_phi_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_phi_t': nn.Embedding(num_ent, t_emb_dim)})

        # frequency embeddings for the entities
        self.embedding.update({'m_amps_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'m_amps_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_amps_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_amps_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_amps_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_amps_t': nn.Embedding(num_ent, t_emb_dim)})

        self.embedding = nn.ModuleDict(self.embedding)

        for k, v in self.embedding.items():
            nn.init.xavier_uniform_(v.weight)

    def get_time_embedding(self, ent, year, month, day, ent_pos):
        # TODO: enum
        if ent_pos == "head":
            time_emb = self.embedding['y_amps_h'](ent) * self.time_nl(
                self.embedding['y_freq_h'](ent) * year + self.embedding['y_phi_h'](ent))
            time_emb += self.embedding['m_amps_h'](ent) * self.time_nl(
                self.embedding['m_freq_h'](ent) * month + self.embedding['m_phi_h'](ent))
            time_emb += self.embedding['d_amps_h'](ent) * self.time_nl(
                self.embedding['d_freq_h'](ent) * day + self.embedding['d_phi_h'](ent))
        else:
            time_emb = self.embedding['y_amps_t'](ent) * self.time_nl(
                self.embedding['y_freq_t'](ent) * year + self.embedding['y_phi_t'](ent))
            time_emb += self.embedding['m_amps_t'](ent) * self.time_nl(
                self.embedding['m_freq_t'](ent) * month + self.embedding['m_phi_t'](ent))
            time_emb += self.embedding['d_amps_t'](ent) * self.time_nl(
                self.embedding['d_freq_t'](ent) * day + self.embedding['d_phi_t'](ent))

        return time_emb

    def get_embedding(self, head, rel, tail, year, month, day):
        year = year.view(-1, 1)
        month = month.view(-1, 1)
        day = day.view(-1, 1)

        h_emb1 = self.embedding['ent_embs_h'](head)
        r_emb1 = self.embedding['rel_embs_f'](rel)
        t_emb1 = self.embedding['ent_embs_t'](tail)

        h_emb2 = self.embedding['ent_embs_h'](tail)
        r_emb2 = self.embedding['rel_embs_i'](rel)
        t_emb2 = self.embedding['ent_embs_t'](head)

        h_emb1 = torch.cat((h_emb1, self.get_time_embedding(head, year, month, day, 'head')), 1)
        t_emb1 = torch.cat((t_emb1, self.get_time_embedding(tail, year, month, day, 'tail')), 1)
        h_emb2 = torch.cat((h_emb2, self.get_time_embedding(tail, year, month, day, 'head')), 1)
        t_emb2 = torch.cat((t_emb2, self.get_time_embedding(head, year, month, day, 'tail')), 1)

        return h_emb1, r_emb1, t_emb1, h_emb2, r_emb2, t_emb2

    @forward_checking
    def forward(self, samples, **kwargs):
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()
        year = samples[:, 3]
        month = samples[:, 4]
        day = samples[:, 5]

        h_emb1, r_emb1, t_emb1, h_emb2, r_emb2, t_emb2 = self.get_embedding(head, rel, tail, year, month, day)

        p = self.config.get('model.dropout')

        scores = ((h_emb1 * r_emb1) * t_emb1 + (h_emb2 * r_emb2) * t_emb2) / 2.0
        scores = F.dropout(scores, p=p, training=self.training)  # TODO training
        scores = torch.sum(scores, dim=1)

        return scores, None

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors

    def predict(self, queries: torch.Tensor):
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores


@BaseModel.register(name="tcomplex")
class TComplExModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        self.rank = self.config.get("model.rank")
        self.no_time_emb = self.config.get("model.no_time_emb")
        self.init_size = self.config.get("model.init_size")

        self.num_ent = self.dataset.num_entities()
        self.num_rel = self.dataset.num_relations()
        self.num_ts = self.dataset.num_timestamps()

        self.prepare_embedding()

    def prepare_embedding(self):
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in [self.num_ent, self.num_rel, self.num_ts]
        ])

        for emb in self.embeddings:
            emb.weight.data *= self.init_size

    @forward_checking
    def forward(self, x):
        """
        x is spot
        """
        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        rhs = self.embeddings[0](x[:, 2].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight  # all ent tensor
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        # 1st item: scores
        # 2nd item: reg item factors
        # 3rd item: time

        scores = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + \
                 (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        factors = {
            "n3": (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)),
            "lambda3": (self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight)
        }

        return scores, factors

    def fit(self, samples: torch.Tensor):
        self.config.assert_true(self.config.get("negative_sampling.type") == 'pseudo_sampling',
                                "Use pseudo_sampling for tcomplex model.")

        bs = samples.size(0)
        dim = samples.size(1)

        samples = samples.view(-1, dim)
        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors

    def predict(self, x):
        self.config.assert_true(torch.isnan(x).sum(1).byte().all(), "Either head or tail should be absent.")

        missing_head_ind = torch.isnan(x)[:, 0].byte().unsqueeze(1)
        reversed_x = x.clone()
        reversed_x[:, 1] += 1
        reversed_x[:, (0, 2)] = reversed_x[:, (2, 0)]

        x = torch.where(missing_head_ind,
                        reversed_x,
                        x)

        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        scores = (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
                  lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) @ right[0].t() + \
                 (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
                  lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) @ right[1].t()

        return scores

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )


@BaseModel.register(name="tntcomplex")
class TNTComplExModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        self.rank = self.config.get("model.rank")
        self.no_time_emb = self.config.get("model.no_time_emb")
        self.init_size = self.config.get("model.init_size")

        self.num_ent = self.dataset.num_entities()
        self.num_rel = self.dataset.num_relations()
        self.num_ts = self.dataset.num_timestamps()

        self.prepare_embedding()

    def prepare_embedding(self):
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in [self.num_ent, self.num_rel, self.num_ts, self.num_rel]
        ])

        for emb in self.embeddings:
            emb.weight.data *= self.init_size

    @forward_checking
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        rel_no_time = self.embeddings[3](x[:, 1].long())
        rhs = self.embeddings[0](x[:, 2].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight  # all ent tensor
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        scores = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + \
                 (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()

        factors = {
            "n3": (math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
                   torch.sqrt(rnt[0] ** 2 + rrt[1] ** 2),
                   math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)),
            "lambda3": (self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight)
        }

        return scores, factors

    def fit(self, samples: torch.Tensor):
        self.config.assert_true(self.config.get("negative_sampling.type") == 'pseudo_sampling',
                                "Use pseudo_sampling for tntcomplex model.")

        bs = samples.size(0)
        dim = samples.size(1)

        samples = samples.view(-1, dim)
        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors

    def predict(self, x):
        self.config.assert_true(torch.isnan(x).sum(1).byte().all(), "Either head or tail should be absent.")

        missing_head_ind = torch.isnan(x)[:, 0].byte().unsqueeze(1)
        reversed_x = x.clone()
        reversed_x[:, 1] += 1  # dangerous
        reversed_x[:, (0, 2)] = reversed_x[:, (2, 0)]

        x = torch.where(missing_head_ind,
                        reversed_x,
                        x)

        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        rel_no_time = self.embeddings[3](x[:, 1].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        scores = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + \
                 (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()

        return scores


@BaseModel.register(name="hyte")
class HyTEModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

    @forward_checking
    def forward(self, samples: torch.Tensor, **kwargs):
        # TODO remember to negate the scores with torch.neg(scores)
        raise NotImplementedError


@BaseModel.register(name="atise")
class ATiSEModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        # TODO(gengyuan) load params before initialize
        self.cmin = self.config.get("model.cmin")
        self.cmax = self.config.get("model.cmax")
        self.emb_dim = self.config.get("model.embedding_dim")

        self.prepare_embedding()

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()

        self.embedding: Dict[str, nn.Module] = defaultdict(None)

        self.embedding.update({'emb_E': nn.Embedding(num_ent, self.emb_dim, padding_idx=0)})
        self.embedding.update({'emb_E_var': nn.Embedding(num_ent, self.emb_dim, padding_idx=0)})
        self.embedding.update({'emb_R': nn.Embedding(num_rel, self.emb_dim, padding_idx=0)})
        self.embedding.update({'emb_R_var': nn.Embedding(num_rel, self.emb_dim, padding_idx=0)})

        self.embedding.update({'emb_TE': nn.Embedding(num_ent, self.emb_dim, padding_idx=0)})
        self.embedding.update({'alpha_E': nn.Embedding(num_ent, 1, padding_idx=0)})
        self.embedding.update({'beta_E': nn.Embedding(num_ent, self.emb_dim, padding_idx=0)})
        self.embedding.update({'omega_E': nn.Embedding(num_ent, self.emb_dim, padding_idx=0)})

        self.embedding.update({'emb_TR': nn.Embedding(num_rel, self.emb_dim, padding_idx=0)})
        self.embedding.update({'alpha_R': nn.Embedding(num_rel, 1, padding_idx=0)})
        self.embedding.update({'beta_R': nn.Embedding(num_rel, self.emb_dim, padding_idx=0)})
        self.embedding.update({'omega_R': nn.Embedding(num_rel, self.emb_dim, padding_idx=0)})

        self.embedding = nn.ModuleDict(self.embedding)

        r = 6 / np.sqrt(self.emb_dim)
        self.embedding['emb_E'].weight.data.uniform_(-r, r)
        self.embedding['emb_E_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_R'].weight.data.uniform_(-r, r)
        self.embedding['emb_R_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_TE'].weight.data.uniform_(-r, r)
        self.embedding['alpha_E'].weight.data.uniform_(0, 0)
        self.embedding['beta_E'].weight.data.uniform_(0, 0)
        self.embedding['omega_E'].weight.data.uniform_(-r, r)
        self.embedding['emb_TR'].weight.data.uniform_(-r, r)
        self.embedding['alpha_R'].weight.data.uniform_(0, 0)
        self.embedding['beta_R'].weight.data.uniform_(0, 0)
        self.embedding['omega_R'].weight.data.uniform_(-r, r)

        self.embedding['emb_E'].weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.embedding['emb_E_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_R'].weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.embedding['emb_R_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_TE'].weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.embedding['emb_TR'].weight.data.renorm_(p=2, dim=0, maxnorm=1)

    @forward_checking
    def forward(self, sample: torch.Tensor):
        # bs = sample.size(0)
        # # TODO(gengyuan)
        # dim = sample.size(1) // (1 + self.config.get("negative_sampling.num_samples"))
        # sample = sample.view(-1, dim)

        # TODO(gengyuan) type conversion when feeding the data instead of running the models
        h_i, t_i, r_i, d_i = sample[:, 0].long(), sample[:, 2].long(), sample[:, 1].long(), sample[:, 3]

        pi = 3.14159265358979323846

        h_mean = self.embedding['emb_E'](h_i).view(-1, self.emb_dim) + \
                 d_i.view(-1, 1) * self.embedding['alpha_E'](h_i).view(-1, 1) * self.embedding['emb_TE'](h_i).view(-1,
                                                                                                                   self.emb_dim) \
                 + self.embedding['beta_E'](h_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.embedding['omega_E'](h_i).view(-1, self.emb_dim) * d_i.view(-1, 1))

        t_mean = self.embedding['emb_E'](t_i).view(-1, self.emb_dim) + \
                 d_i.view(-1, 1) * self.embedding['alpha_E'](t_i).view(-1, 1) * self.embedding['emb_TE'](t_i).view(-1,
                                                                                                                   self.emb_dim) \
                 + self.embedding['beta_E'](t_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.embedding['omega_E'](t_i).view(-1, self.emb_dim) * d_i.view(-1, 1))

        r_mean = self.embedding['emb_R'](r_i).view(-1, self.emb_dim) + \
                 d_i.view(-1, 1) * self.embedding['alpha_R'](r_i).view(-1, 1) * self.embedding['emb_TR'](r_i).view(-1,
                                                                                                                   self.emb_dim) \
                 + self.embedding['beta_R'](r_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.embedding['omega_R'](r_i).view(-1, self.emb_dim) * d_i.view(-1, 1))

        h_var = self.embedding['emb_E_var'](h_i).view(-1, self.emb_dim)
        t_var = self.embedding['emb_E_var'](t_i).view(-1, self.emb_dim)
        r_var = self.embedding['emb_R_var'](r_i).view(-1, self.emb_dim)

        out1 = torch.sum((h_var + t_var) / r_var, 1) + torch.sum(((r_mean - h_mean + t_mean) ** 2) / r_var,
                                                                 1) - self.emb_dim
        out2 = torch.sum(r_var / (h_var + t_var), 1) + torch.sum(((h_mean - t_mean - r_mean) ** 2) / (h_var + t_var),
                                                                 1) - self.emb_dim
        scores = (out1 + out2) / 4

        factors = {
            "renorm": (self.embedding['emb_E'].weight,
                       self.embedding['emb_R'].weight,
                       self.embedding['emb_TE'].weight,
                       self.embedding['emb_TR'].weight),
            "clamp": (self.embedding['emb_E_var'].weight,
                      self.embedding['emb_R_var'].weight)
        }

        return scores, factors

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)
        scores, factors = self.forward(samples)

        scores = scores.view(bs, -1)

        return scores, factors

    def predict(self, queries: torch.Tensor):
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores


# reference: https://github.com/bsantraigi/TA_TransE/blob/master/model.py
# reference: https://github.com/jimmywangheng/knowledge_representation_pytorch
@BaseModel.register(name="ta_transe")
class TATransEModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        # model params from files
        self.emb_dim = self.config.get("model.emb_dim")
        self.l1_flag = self.config.get("model.l1_flag")
        self.p = self.config.get("model.p")

        self.dropout = torch.nn.Dropout(p=self.p)
        self.lstm = LSTMModel(self.emb_dim, n_layer=1)

        self.prepare_embedding()

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()
        num_tem = 32  # should be 32

        self.embedding: Dict[str, torch.nn.Embedding] = defaultdict(None)
        self.embedding['ent'] = torch.nn.Embedding(num_ent, self.emb_dim)
        self.embedding['rel'] = torch.nn.Embedding(num_rel, self.emb_dim)
        self.embedding['tem'] = torch.nn.Embedding(num_tem, self.emb_dim)

        self.embedding = nn.ModuleDict(self.embedding)

        for _, emb in self.embedding.items():
            torch.nn.init.xavier_uniform_(emb.weight)
            emb.weight.data.renorm(p=2, dim=1, maxnorm=1)

    def get_rseq(self, rel: torch.LongTensor, tem: torch.LongTensor):
        r_e = self.embedding['rel'](rel)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        bs = tem.size(0)
        tem_len = tem.size(1)
        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)

        token_e = self.embedding['tem'](tem)
        token_e = token_e.view(bs, tem_len, self.emb_dim)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return rseq_e

    @forward_checking
    def forward(self, samples: torch.Tensor):
        torch.cuda.empty_cache()
        h, r, t, tem = samples[:, 0].long(), samples[:, 1].long(), samples[:, 2].long(), samples[:, 3:].long()

        h_e = self.embedding['ent'](h)
        t_e = self.embedding['ent'](t)
        rseq_e = self.get_rseq(r, tem)

        h_e = self.dropout(h_e)
        t_e = self.dropout(t_e)
        rseq_e = self.dropout(rseq_e)

        if self.l1_flag:
            scores = torch.neg(torch.sum(torch.abs(h_e + rseq_e - t_e), 1))
        else:
            scores = torch.neg(torch.sum((h_e + rseq_e - t_e) ** 2, 1))

        # factors = {
        #     "norm": (h_e,
        #              t_e,
        #              rseq_e)
        # }

        # factors = {"entity_reg": self.embedding['ent'],
        #            "relation_reg": self.embedding['rel']
        #            }

        factors = None

        return scores, factors

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor

    def predict(self, queries: torch.Tensor):
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores


# reference: https://github.com/bsantraigi/TA_TransE/blob/master/model.py
@BaseModel.register(name="ta_distmult")
class TADistmultModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        # model params from files
        self.emb_dim = self.config.get("model.emb_dim")
        self.l1_flag = self.config.get("model.l1_flag")
        self.p = self.config.get("model.p")

        self.dropout = torch.nn.Dropout(p=self.p)
        self.lstm = LSTMModel(self.emb_dim, n_layer=1)
        self.criterion = nn.Softplus()

        self.prepare_embedding()

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()
        num_tem = 32  # should be 32

        self.embedding: Dict[str, torch.nn.Embedding] = defaultdict(None)
        self.embedding['ent'] = torch.nn.Embedding(num_ent, self.emb_dim)
        self.embedding['rel'] = torch.nn.Embedding(num_rel, self.emb_dim)
        self.embedding['tem'] = torch.nn.Embedding(num_tem, self.emb_dim)

        self.embedding = nn.ModuleDict(self.embedding)

        for _, emb in self.embedding.items():
            torch.nn.init.xavier_uniform_(emb.weight)
            emb.weight.data.renorm(p=2, dim=1, maxnorm=1)

    @forward_checking
    def forward(self, samples: torch.Tensor):
        torch.cuda.empty_cache()
        h, r, t, tem = samples[:, 0].long(), samples[:, 1].long(), samples[:, 2].long(), samples[:, 3:].long()

        h_e = self.embedding['ent'](h)
        t_e = self.embedding['ent'](t)
        rseq_e = self.get_rseq(r, tem)

        h_e = self.dropout(h_e)
        t_e = self.dropout(t_e)
        rseq_e = self.dropout(rseq_e)

        scores = torch.sum(h_e * t_e * rseq_e, 1, False)

        factors = {
            "norm": (self.embedding['ent'].weight,
                     self.embedding['rel'].weight,
                     self.embedding['tem'].weight)
        }

        return scores, factors

    def get_rseq(self, rel, tem):
        r_e = self.embedding['rel'](rel)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        bs = tem.size(0)
        tem_len = tem.size(1)
        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)

        token_e = self.embedding['tem'](tem)
        token_e = token_e.view(bs, tem_len, self.emb_dim)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return rseq_e

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor

    def predict(self, queries: torch.Tensor):
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores


@BaseModel.register(name="ttranse")
class TTransEModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        # model params from files
        self.emb_dim = self.config.get("model.emb_dim")
        self.l1_flag = self.config.get("model.l1_flag")

        self.prepare_embedding()

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()
        num_tem = self.dataset.num_timestamps()

        self.embedding: Dict[str, torch.nn.Embedding] = defaultdict(None)
        self.embedding['ent'] = torch.nn.Embedding(num_ent, self.emb_dim)
        self.embedding['rel'] = torch.nn.Embedding(num_rel, self.emb_dim)
        self.embedding['tem'] = torch.nn.Embedding(num_tem, self.emb_dim)

        self.embedding = nn.ModuleDict(self.embedding)

        for _, emb in self.embedding.items():
            torch.nn.init.xavier_uniform_(emb.weight)
            emb.weight.data.renorm(p=2, dim=1, maxnorm=1)

    @forward_checking
    def forward(self, samples, **kwargs):
        h, r, t, tem = samples[:, 0].long(), samples[:, 1].long(), samples[:, 2].long(), samples[:, 3].long()

        h_e = self.embedding['ent'](h)
        t_e = self.embedding['ent'](t)
        r_e = self.embedding['rel'](r)
        tem_e = self.embedding['tem'](tem)

        if self.l1_flag:
            scores = torch.neg(torch.sum(torch.abs(h_e + r_e + tem_e - t_e), dim=1))
        else:
            scores = torch.neg(torch.sum((h_e + r_e + tem_e - t_e) ** 2, dim=1))

        factors = {
            "norm": (h_e,
                     t_e,
                     r_e,
                     tem_e)
        }

        return scores, factors

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor

    def predict(self, queries: torch.Tensor):
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores


@BaseModel.register(name="tcomplex_tango")
class TComplExTangoModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        self.rank = self.config.get("model.rank")
        self.no_time_emb = self.config.get("model.no_time_emb")
        self.init_size = self.config.get("model.init_size")

        self.num_ent = self.dataset.num_entities()
        self.num_rel = self.dataset.num_relations()
        self.num_ts = self.dataset.num_timestamps()

        self.prepare_embedding()

    def prepare_embedding(self):
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in [self.num_ent, self.num_rel, self.num_ts]
        ])

        for emb in self.embeddings:
            emb.weight.data *= self.init_size

    @forward_checking
    def forward(self, x):
        """
        x is spot
        """
        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        rhs = self.embeddings[0](x[:, 2].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight  # all ent tensor
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        # 1st item: scores
        # 2nd item: reg item factors
        # 3rd item: time

        scores = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + \
                 (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        factors = {
            "n3": (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)),
            "lambda3": (self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight)
        }

        return scores, factors

    def fit(self, samples: torch.Tensor):
        self.config.assert_true(self.config.get("negative_sampling.type") == 'pseudo_sampling',
                                "Use pseudo_sampling for tcomplex model.")

        bs = samples.size(0)
        dim = samples.size(1)

        samples = samples.view(-1, dim)
        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors

    def predict(self, x):
        self.config.assert_true(torch.isnan(x).sum(1).byte().all(), "Either head or tail should be absent.")

        missing_head_ind = torch.isnan(x)[:, 0].byte().unsqueeze(1)
        reversed_x = x.clone()
        reversed_x[:, 1] += self.dataset.num_relations() // 2
        reversed_x[:, (0, 2)] = reversed_x[:, (2, 0)]

        x = torch.where(missing_head_ind,
                        reversed_x,
                        x)

        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        scores = (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
                  lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) @ right[0].t() + \
                 (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
                  lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) @ right[1].t()

        return scores

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )


@BaseModel.register(name="tntcomplex_tango")
class TNTComplExTangoModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        self.rank = self.config.get("model.rank")
        self.no_time_emb = self.config.get("model.no_time_emb")
        self.init_size = self.config.get("model.init_size")

        self.num_ent = self.dataset.num_entities()
        self.num_rel = self.dataset.num_relations()
        self.num_ts = self.dataset.num_timestamps()

        self.prepare_embedding()

    def prepare_embedding(self):
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in [self.num_ent, self.num_rel, self.num_ts, self.num_rel]
        ])

        for emb in self.embeddings:
            emb.weight.data *= self.init_size

    @forward_checking
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        rel_no_time = self.embeddings[3](x[:, 1].long())
        rhs = self.embeddings[0](x[:, 2].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight  # all ent tensor
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        scores = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + \
                 (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()

        factors = {
            "n3": (math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
                   torch.sqrt(rnt[0] ** 2 + rrt[1] ** 2),
                   math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)),
            "lambda3": (self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight)
        }

        return scores, factors

    def fit(self, samples: torch.Tensor):
        self.config.assert_true(self.config.get("negative_sampling.type") == 'pseudo_sampling',
                                "Use pseudo_sampling for tntcomplex model.")

        bs = samples.size(0)
        dim = samples.size(1)

        samples = samples.view(-1, dim)
        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors

    def predict(self, x):
        self.config.assert_true(torch.isnan(x).sum(1).byte().all(), "Either head or tail should be absent.")

        missing_head_ind = torch.isnan(x)[:, 0].byte().unsqueeze(1)
        reversed_x = x.clone()
        reversed_x[:, 1] += self.dataset.num_relations() // 2  # dangerous
        reversed_x[:, (0, 2)] = reversed_x[:, (2, 0)]

        x = torch.where(missing_head_ind,
                        reversed_x,
                        x)

        lhs = self.embeddings[0](x[:, 0].long())
        rel = self.embeddings[1](x[:, 1].long())
        rel_no_time = self.embeddings[3](x[:, 1].long())
        time = self.embeddings[2](x[:, 3].long())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        scores = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + \
                 (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()

        return scores


@BaseModel.register(name="ConT")
class ConTModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor, **kwargs):
        super().__init__(config, dataset)

        # model params from files
        self.emb_dim = self.config.get("model.emb_dim")

        self.prepare_embedding()

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()
        num_tem = self.dataset.num_timestamps()

        self.embedding: Dict[str, torch.nn.Embedding] = defaultdict(None)
        self.embedding['ent_head'] = torch.nn.Embedding(num_ent, self.emb_dim)
        self.embedding['ent_tail'] = torch.nn.Embedding(num_ent, self.emb_dim)
        self.embedding['rel'] = torch.nn.Embedding(num_rel, self.emb_dim)

        self.g = nn.Parameter(torch.zeros(num_tem, self.emb_dim, self.emb_dim, self.emb_dim), requires_grad=True)

        self.embedding = nn.ModuleDict(self.embedding)

        for _, emb in self.embedding.items():
            torch.nn.init.xavier_uniform_(emb.weight)
            emb.weight.data.renorm(p=2, dim=1, maxnorm=1)

        torch.nn.init.xavier_uniform_(self.g.data)

    @forward_checking
    def forward(self, samples, **kwargs):
        h, r, t, tem = samples[:, 0].long(), samples[:, 1].long(), samples[:, 2].long(), samples[:, 3].long()

        h_e = self.embedding['ent_head'](h)  # 1d vector
        t_e = self.embedding['ent_tail'](t)  # 1d vector
        r_e = self.embedding['rel'](r)  # 1d vector

        g_t = self.g[tem, ...]  # 3D tensor

        scores = torch.sum(g_t * h_e.view(-1, self.emb_dim, 1, 1), dim=1)  # 2D
        scores = torch.sum(scores * r_e.view(-1, self.emb_dim, 1), dim=1)  # 1D
        scores = torch.sum(scores * t_e.view(-1, self.emb_dim), dim=1)

        factors = {
            "L2": (h_e,
                   t_e,
                   r_e,
                   g_t)
        }

        return scores, factors

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor

    def predict(self, queries: torch.Tensor):
        self.config.assert_true(torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent.")

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores
