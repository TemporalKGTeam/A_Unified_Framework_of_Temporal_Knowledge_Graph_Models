import torch
from torch.utils.data.dataset import Dataset as PTDataset
import numpy as np

from typing import Dict, List, Tuple, Optional
import enum
import datetime
import time

from tkge.data.dataset import DatasetProcessor
from tkge.common.config import Config
from tkge.data.utils import get_all_days_between
import arrow

from collections import defaultdict

SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))


@DatasetProcessor.register(name="icews14_atise")
class ICEWS14AtiseDatasetProcessor(DatasetProcessor):
    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        # TODO (gengyuan) move to init method
        self.gran = self.config.get("dataset.temporal.gran")

        start_sec = time.mktime(time.strptime('2014-01-01', '%Y-%m-%d'))

        end_sec = time.mktime(time.strptime(origin, '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (self.gran * 24 * 60 * 60))

        return day


@DatasetProcessor.register(name="icews11-14_atise")
class ICEWS1114AtiseDatasetProcessor(DatasetProcessor):
    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        # TODO (gengyuan) move to init method
        self.gran = self.config.get("dataset.temporal.gran")

        start_sec = time.mktime(time.strptime('2011-01-01', '%Y-%m-%d'))

        end_sec = time.mktime(time.strptime(origin, '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (self.gran * 24 * 60 * 60))

        return day


@DatasetProcessor.register(name="gdelt-m10-atise")
class GDELTM10AtiseDatasetProcessor(DatasetProcessor):
    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        # TODO (gengyuan) move to init method
        self.gran = self.config.get("dataset.temporal.gran")

        start_sec = time.mktime(time.strptime('2015-10-01', '%Y-%m-%d'))

        end_sec = time.mktime(time.strptime(origin, '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (self.gran * 24 * 60 * 60))

        return day


@DatasetProcessor.register(name="icews14_TA")
class ICEWS14TADatasetProcessor(DatasetProcessor):
    def process(self):
        self.tem_dict = {
            '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
            '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18,
            '10m': 19, '11m': 20, '12m': 21,
            '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
        }

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        ts = []
        year, month, day = origin.split('-')

        ts.extend([self.tem_dict[f'{int(yi):01}y'] for yi in year])
        ts.extend([self.tem_dict[f'{int(month):02}m']])
        ts.extend([self.tem_dict[f'{int(di):01}d'] for di in day])

        return ts

    def num_time_identifier(self):
        return 32


# Deprecated: dataset used for debugging tcomplex training
@DatasetProcessor.register(name="icews14_tcomplex")
class TestICEWS14DatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

        print('==========')

        self.folder = self.config.get("dataset.folder")
        self.level = self.config.get("dataset.temporal.resolution")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")

        self.reciprocal_training = self.config.get("task.reciprocal_relation")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw: List[str] = []
        self.valid_raw: List[str] = []
        self.test_raw: List[str] = []

        mapping = torch.load('/mnt/data1/ma/gengyuan/baseline/tkbc/tkbc/mapping.pt')

        self.ent2id = mapping[0]
        self.rel2id = mapping[1]
        self.ts2id = mapping[2]

        temp = dict()

        for k in self.rel2id.keys():
            temp[k + '(RECIPROCAL)'] = self.rel2id[k] + 230

        self.rel2id.update(temp)

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.load()
        self.process()
        self.filter()

    def load(self):
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        with open(train_file, "r") as f:
            if self.reciprocal_training:
                lines = f.readlines()

                for line in lines:
                    self.train_raw.append(line)

                    # for line in lines:

                    insert_line = line.split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.train_raw.append(insert_line)
            else:
                self.train_raw = f.readlines()

            self.train_size = len(self.train_raw)

        with open(valid_file, "r") as f:
            self.valid_raw = f.readlines()

            self.valid_size = len(self.valid_raw)

        with open(test_file, "r") as f:
            self.test_raw = f.readlines()

            self.test_size = len(self.test_raw)

    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        level = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(self.level in level, f"Time granularity should be {level}")

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:level.index(self.level) + 1]
        ts = '-'.join(ts)

        return ts

    # def index_relations(self, rel: str):
    #     if rel.endswith('(RECIPROCAL)'):
    #         print(rel)
    #         return self.rel2id[rel[:-12]] + 230
    #     else:
    #         return self.rel2id[rel]


@DatasetProcessor.register(name="yago189")
class YAGO1830DatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

        self.folder = self.config.get("dataset.folder")
        self.level = self.config.get("dataset.temporal.resolution")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")

        self.reciprocal_training = self.config.get("task.reciprocal_training")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw: List[str] = []
        self.valid_raw: List[str] = []
        self.test_raw: List[str] = []

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.load()
        self.process()
        self.filter()

    def load(self):
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        with open(train_file, "r") as f:
            if self.reciprocal_training:
                lines = f.readlines()

                for line in lines:
                    self.train_raw.append(line)

                    # for line in lines:

                    insert_line = line.split('\t')
                    insert_line[1] = str(int(insert_line[1]) + 1)
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.train_raw.append(insert_line)
            else:
                self.train_raw = f.readlines()

            self.train_size = len(self.train_raw)

        with open(valid_file, "r") as f:
            self.valid_raw = f.readlines()

            self.valid_size = len(self.valid_raw)

        with open(test_file, "r") as f:
            self.test_raw = f.readlines()

            self.test_size = len(self.test_raw)

    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts, _ = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = int(ts)
            ts_id = ts

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts, _ = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = int(ts)
            ts_id = ts

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts, _ = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = int(ts)
            ts_id = ts

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        raise NotImplementedError

    def num_entities(self):
        return 10623

    def num_relations(self):
        return 10

    def num_timestamps(self):
        return 189


@DatasetProcessor.register(name="wiki_ding")
class WikiDingDatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

        self.folder = self.config.get("dataset.folder")
        self.level = self.config.get("dataset.temporal.resolution")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")

        self.reciprocal_training = self.config.get("task.reciprocal_training")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw: List[str] = []
        self.valid_raw: List[str] = []
        self.test_raw: List[str] = []

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.load()
        self.process()
        self.filter()

    def load(self):
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        with open(train_file, "r") as f:
            if self.reciprocal_training:
                lines = f.readlines()

                for line in lines:
                    self.train_raw.append(line)

                    # for line in lines:

                    insert_line = line.split('\t')
                    insert_line[1] = str(int(insert_line[1]) + 24)
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.train_raw.append(insert_line)
            else:
                self.train_raw = f.readlines()

            self.train_size = len(self.train_raw)

        with open(valid_file, "r") as f:
            self.valid_raw = f.readlines()

            self.valid_size = len(self.valid_raw)

        with open(test_file, "r") as f:
            self.test_raw = f.readlines()

            self.test_size = len(self.test_raw)

    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts, _ = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = int(ts)
            ts_id = ts

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts, _ = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = int(ts)
            ts_id = ts

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts, _ = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = int(ts)
            ts_id = ts

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        raise NotImplementedError

    def num_entities(self):
        return 12554

    def num_relations(self):
        return 24

    def num_timestamps(self):
        return 232


@DatasetProcessor.register(name="han")
class HANDatasetProcessor(DatasetProcessor):
    def process(self):
        self.ts2id = {str(i): i for i in range(189)}   #232 189

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.valid_set['triple'].append([head, rel, tail])
                self.valid_set['timestamp_id'].append([ts_id])
                self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.test_set['triple'].append([head, rel, tail])
                self.test_set['timestamp_id'].append([ts_id])
                self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):

        return origin


@DatasetProcessor.register(name="gdelt-m10_TA")
class GDELTM10TADatasetProcessor(DatasetProcessor):
    def process(self):
        self.tem_dict = {
            '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
            '10m': 10,
            '0d': 11, '1d': 12, '2d': 13, '3d': 14, '4d': 15, '5d': 16, '6d': 17, '7d': 18, '8d': 19, '9d': 20,
        }
        all_timestamp = get_all_days_between('2015-10-01', '2015-10-31')
        self.ts2id = {ts: (arrow.get(ts) - arrow.get('2015-10-01')).days for ts in all_timestamp}

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            if '(RECIPROCAL)' not in rd:
                self.valid_set['triple'].append([head, rel, tail])
                self.valid_set['timestamp_id'].append([ts_id])
                self.valid_set['timestamp_float'].append(ts)

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            if '(RECIPROCAL)' not in rd:
                self.test_set['triple'].append([head, rel, tail])
                self.test_set['timestamp_id'].append([ts_id])
                self.test_set['timestamp_float'].append(ts)

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        ts = []
        year, month, day = origin.split('-')

        ts.extend([self.tem_dict[f'{int(yi):01}y'] for yi in year])
        ts.extend([self.tem_dict[f'{int(month):02}m']])
        ts.extend([self.tem_dict[f'{int(di):01}d'] for di in day])

        return ts

    def num_time_identifier(self):
        return 21