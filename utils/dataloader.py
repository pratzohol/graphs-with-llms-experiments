import torch
import os
import os.path as osp
import sys
import random
from torch.utils.data import Sampler


sys.path.append(osp.join(osp.dirname(__file__), ".."))

from utils.cora import CoraPyGDataset
from utils.pubmed import PubmedPyGDataset
from utils.ogbn_arxiv import ArxivPyGDataset
from utils.ogbn_products import ProductsPyGDataset
from utils.encoder import SentenceEncoder
from utils.task_constructor import *


class BatchSampler(Sampler):
    def __init__(self, batch_count, batch_size, task, seed=None):
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.task = task
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.batch_count):
            yield self.sample()

    def __len__(self):
        return self.batch_count

    def sample(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.task.sample())
        return batch


class GetDataloader:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs["dataset"].lower()
        self.dataRoot = kwargs["dataRoot"]
        self.custom_dataRoot = kwargs["custom_dataRoot"]

        self.sentence_encoder = SentenceEncoder(name=kwargs["sentence_encoder"], root=kwargs["encoder_path"], device=kwargs["device"])

        if self.dataset_name == "cora":
            self.custom_data = CoraPyGDataset(dataRoot=self.dataRoot, custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)

        elif self.dataset_name == "pubmed":
            self.custom_data = PubmedPyGDataset(dataRoot=self.dataRoot, custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)

        elif self.dataset_name in ["ogbn_arxiv", "arxiv"]:
            self.custom_data = ArxivPyGDataset(dataRoot=self.dataRoot, custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)

        elif self.dataset_name in ["ogbn_products", "products"]:
            self.custom_data = ProductsPyGDataset(dataRoot=self.dataRoot, custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)

        else:
            raise NotImplementedError

    def get_data(self):
        return self.custom_data._data


if __name__ == '__main__':
