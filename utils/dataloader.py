import torch
import os
import os.path as osp
import sys
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

sys.path.append(osp.join(osp.dirname(__file__), ".."))

from utils_data.cora import CoraPyGDataset
from utils_data.pubmed import PubmedPyGDataset
from utils_data.ogbn_arxiv import ArxivPyGDataset
from utils_data.ogbn_products import ProductsPyGDataset
from utils.encoder import SentenceEncoder
from utils.task_constructor import TaskConstructor
from utils.sampler import BatchSampler


class Collator:
    def __init__(self, params):
        pass

    def __call__(self):
        pass


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

        smplr_params = {
            "batch_count": kwargs["batch_count"],
            "batch_size": kwargs["batch_size"],
            "seed": kwargs["seed"],
            # TODO: Add below params to arg.py and config.yaml
            "n_way": kwargs["n_way"],
            "n_shot": kwargs["n_shot"],
            "n_query": kwargs["n_query"],
            "n_member" : kwargs["n_shot"] + kwargs["n_query"],
        }

        self.trn_smplr = BatchSampler(params=smplr_params, task=TaskConstructor(self.custom_data, split="train_mask"))
        self.val_smplr = BatchSampler(params=smplr_params, task=TaskConstructor(self.custom_data, split="val_mask"))
        self.test_smplr = BatchSampler(params=smplr_params, task=TaskConstructor(self.custom_data, split="test_mask"))

        wrkrs = kwargs["num_workers"]
        self.train_dataloader = DataLoader(self.custom_data, batch_sampler=self.trn_smplr, num_workers=wrkrs, collate_fn=Collator(params=smplr_params))
        self.val_dataloader = DataLoader(self.custom_data, batch_sampler=self.val_smplr, num_workers=wrkrs, collate_fn=Collator(params=smplr_params))
        self.test_dataloader = DataLoader(self.custom_data, batch_sampler=self.test_smplr, num_workers=wrkrs, collate_fn=Collator(params=smplr_params))


    def get_dataloader(self):
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def get_num_classes(self):
        return self.custom_data.num_classes

    def get_data(self):
        return self.custom_data._data


if __name__ == '__main__':
    pass