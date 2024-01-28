import torch
from utils.cora import CoraPyGDataset
from utils.pubmed import PubmedPyGDataset
from utils.ogbn_arxiv import ArxivPyGDataset
from utils.ogbn_products import ProductsPyGDataset
from utils.encoder import SentenceEncoder

class GetDataloader:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs["dataset"].lower()
        self.dataRoot = kwargs["dataRoot"]
        self.custom_dataRoot = kwargs["custom_dataRoot"]
        self.device = "cpu" if kwargs["device"] == 123 else f"cuda:{kwargs['device']}"
        self.sentence_encoder = SentenceEncoder(name=kwargs["sentence_encoder"], device=self.device)

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


