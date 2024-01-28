from ogb.nodeproppred import PygNodePropPredDataset
import os
import os.path as osp
import torch_geometric
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import remove_self_loops
import random
from datasets import load_dataset

from utils.encoder import SentenceEncoder


def get_node_feature(data_root):
    nodeidx2asin = pd.read_csv(osp.join(data_root, "ogbn_products/mapping/nodeidx2asin.csv.gz"), index_col="node idx")

    raw_train = load_dataset("json", data_files=osp.join(data_root, "ogbn_products/Amazon-3M.raw/trn.json.gz"))
    raw_test = load_dataset("json", data_files=osp.join(data_root, "ogbn_products/Amazon-3M.raw/tst.json.gz"))

    raw_train_df = raw_train["train"].to_pandas()
    raw_test_df = raw_test["train"].to_pandas()
    raw_combined_df = pd.concat([raw_train_df, raw_test_df], ignore_index=True)

    products_titdesc = pd.merge(nodeidx2asin, raw_combined_df, left_on="asin", right_on="uid")

    # Prompt for the feature of nodes (can be changed accordingly)
    node_feature_prompt = ("Feature Node.\n"
                           + "Product Title and Description : "
                           + products_titdesc["title"]
                           + " + "
                           + products_titdesc["content"])

    node_feature_prompt_list = node_feature_prompt.values
    return node_feature_prompt_list


def get_label_feature(data_root):
    label2cat = pd.read_csv(osp.join(data_root, "ogbn_products/mapping/labelidx2productcategory.csv.gz"), index_col="label idx")

    # Fixing few errors
    label2cat.loc[24] = "Label 25"
    label2cat.loc[45] = "Furniture & Decor"
    label2cat.loc[46] = "Label 47" # replacing '#508510'

    # Prompt for the label nodes (can be changed accordingly)
    label_node_prompt = ("Prompt Node.\n"
                         + "Product Category : "
                         + label2cat["product category"])

    label_node_prompt_list = label_node_prompt.values
    return label_node_prompt_list


class ProductsPyGDataset(InMemoryDataset):
    def __init__(self, dataRoot="../data", custom_dataRoot="../custom_data", sentence_encoder=None, transform=None, pre_transform=None, pre_filter=None):
        self.data_root = dataRoot
        self.custom_data_root = custom_dataRoot
        self.sentence_encoder = sentence_encoder
        self.custom_data_dir = osp.join(self.custom_data_root, f"ogbn_products_{self.sentence_encoder.name}")

        if not osp.exists(self.custom_data_dir):
            os.makedirs(self.custom_data_dir)

        super().__init__(self.custom_data_dir, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt", "texts.pkl"]

    def text_to_embed(self, texts):
        if self.sentence_encoder is None:
            raise NotImplementedError("Sentence Encoder is not passed")
        if texts is None:
            return None
        else:
            return self.sentence_encoder.encode(texts) # returns to self.device

    def encode_texts(self, texts):
        if isinstance(texts[0], str):
            return self.text_to_embed(texts)
        return [self.text_to_embed(t) for t in texts]

    def generate_custom_data(self):
        node_texts = get_node_feature(self.data_root).tolist()
        label_texts = get_label_feature(self.data_root).tolist()

        # Prompt for prompt node/edge and edge texts (can be changed accordingly)
        edge_texts = ["Feature Edge.\n Co-purchased. Two products were purchased together on Amazon"]
        prompt_texts = ["Prompt Node.\n Node Classification of Product Category"]
        prompt_edge_texts = ["Prompt Edge."]

        return [node_texts, label_texts, edge_texts, prompt_texts, prompt_edge_texts]

    def process(self):
        products_data = PygNodePropPredDataset(name="ogbn-products", root=self.data_root)
        products_data_list = products_data._data

        products_data_list.edge_index = remove_self_loops(products_data_list.edge_index)[0] # remove self-loops from graph
        products_data_list.y = products_data_list.y.squeeze()  # to flatten the y tensor

        texts = self.generate_custom_data()
        texts_embed = self.encode_texts(texts)

        torch.save(texts, self.processed_paths[1])

        products_data_list.x_text_feat = texts_embed[0] # node text feature
        products_data_list.label_text_feat = texts_embed[1] # label text feature
        products_data_list.edge_text_feat = texts_embed[2] # edge text feature
        products_data_list.prompt_text_feat = texts_embed[3] # prompt node text feature
        products_data_list.prompt_edge_feat = texts_embed[4] # prompt edge text feature

        # get dataset split
        split_idx = products_data.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # Generate 10 different permutations of train/valid/test split
        train_idx_list = [train_idx[torch.randperm(train_idx.shape[0])] for _ in range(10)]
        valid_idx_list = [valid_idx[torch.randperm(valid_idx.shape[0])] for _ in range(10)]
        test_idx_list = [test_idx[torch.randperm(test_idx.shape[0])] for _ in range(10)]

        products_data_list.train_masks = train_idx_list
        products_data_list.val_masks = valid_idx_list
        products_data_list.test_masks = test_idx_list

        data, slices = self.collate([products_data_list]) # Pass the data_list as a list

        torch.save((data, slices), self.processed_paths[0])
        print("Products is processed. Saved.")
