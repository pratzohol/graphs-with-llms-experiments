import torch
from utils.cora import CoraPyGDataset
from utils.pubmed import PubmedPyGDataset
from utils.ogbn_arxiv import ArxivPyGDataset
from utils.ogbn_products import ProductsPyGDataset
from utils.encoder import SentenceEncoder

class GetDataloader:
    def __init__(self, custom_dataRoot="../custom_data", dataset_name="cora", sentence_encoder="ST", device=2):
        self.dataset_name = dataset_name.lower()
        self.custom_dataRoot = custom_dataRoot
        self.device = "cpu" if device == 123 else f"cuda:{device}"
        self.sentence_encoder = SentenceEncoder(name=sentence_encoder, device=self.device)

        if self.dataset_name == "cora":
            self.custom_data = CoraPyGDataset(custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)._data

        elif self.dataset_name == "pubmed":
            self.custom_data = PubmedPyGDataset(custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)._data

        elif self.dataset_name in ["ogbn_arxiv", "arxiv"]:
            self.custom_data = ArxivPyGDataset(custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)._data

        elif self.dataset_name in ["ogbn_products", "products"]:
            self.custom_data = ProductsPyGDataset(custom_dataRoot=self.custom_dataRoot, sentence_encoder=self.sentence_encoder)._data

        else:
            raise NotImplementedError

    def get_data(self):
        return self.custom_data


