from ogb.nodeproppred import PygNodePropPredDataset
import os
import os.path as osp
import torch_geometric
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url
import random

from utils.encoder import SentenceEncoder


def get_taxonomy(data_root):
    # read categories and description file
    f = open(osp.join(data_root, "ogbn_arxiv", "arxiv_CS_categories.txt"), "r").readlines()

    state = 0
    result = {"id": [], "name": [], "description": []}

    for line in f:
        if state == 0:
            assert line.strip().startswith("cs.")
            category = ("arxiv "
                + " ".join(line.strip().split(" ")[0].split(".")).lower())
            # e.g. cs lo

            name = line.strip()[7:-1]  # e. g. Logic in CS
            result["id"].append(category)
            result["name"].append(name)
            state = 1
            continue

        elif state == 1:
            description = line.strip()
            result["description"].append(description)
            state = 2
            continue

        elif state == 2:
            state = 0
            continue

    arxiv_cs_taxonomy = pd.DataFrame(result)
    return arxiv_cs_taxonomy


def get_node_feature(data_root):
    nodeidx2paperid = pd.read_csv(osp.join(data_root, "ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"), index_col="node idx")

    # Load the title and abstract of each paper
    titleabs_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv"
    titleabs_path = osp.join(data_root, "ogbn_arxiv", "titleabs.tsv")

    if (osp.exists(titleabs_path)):
        titleabs = pd.read_csv(titleabs_path, sep="\t", names=["paper id", "title", "abstract"], index_col="paper id")
    else:
        titleabs = pd.read_csv(titleabs_url, sep="\t", names=["paper id", "title", "abstract"], index_col="paper id")

    titleabs = nodeidx2paperid.join(titleabs, on="paper id")


    # Prompt for the feature of nodes (can be changed accordingly)
    node_feature_prompt = ("Feature Node.\n"
                        + "Paper Title and Abstract : "
                        + titleabs["title"]
                        + " + "
                        + titleabs["abstract"])

    node_feature_prompt_list = node_feature_prompt.values
    return node_feature_prompt_list


def get_label_feature(data_root):
    arxiv_cs_taxonomy = get_taxonomy(data_root)

    mapping_file = osp.join(data_root, "ogbn_arxiv", "mapping", "labelidx2arxivcategeory.csv.gz")
    labelidx2arxivcategory = pd.read_csv(mapping_file)

    arxiv_categ_vals = pd.merge(labelidx2arxivcategory, arxiv_cs_taxonomy, left_on="arxiv category", right_on="id")


    # Prompt for the label nodes (can be changed accordingly)
    label_node_prompt = ("Prompt Node.\n"
                        + "Literature Category and Description: "
                        + arxiv_categ_vals["name"]
                        + " + "
                        + arxiv_categ_vals["description"])

    label_node_prompt_list = label_node_prompt.values
    return label_node_prompt_list


class ArxivPyGDataset(InMemoryDataset):
    def __init__(self, dataRoot="../data", custom_dataRoot="../custom_data", sentence_encoder=None, transform=None, pre_transform=None, pre_filter=None):
        self.data_root = dataRoot
        self.custom_data_root = custom_dataRoot
        self.sentence_encoder = sentence_encoder
        self.custom_data_dir = osp.join(self.custom_data_root, f"ogbn_arxiv_{self.sentence_encoder.name}")

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
        edge_texts = ["Feature Edge.\n Citation"]
        prompt_texts = ["Prompt Node.\n Node Classification of Literature Category"]
        prompt_edge_texts = ["Prompt Edge."]

        return [node_texts, label_texts, edge_texts, prompt_texts, prompt_edge_texts]

    def process(self):
        arxiv_data = PygNodePropPredDataset(name="ogbn-arxiv", root=self.data_root)
        arxiv_data_list = arxiv_data._data

        arxiv_data_list.y = arxiv_data_list.y.squeeze()  # to flatten the y tensor

        texts = self.generate_custom_data()
        texts_embed = self.encode_texts(texts)

        torch.save(texts, self.processed_paths[1])

        arxiv_data_list.x_text_feat = texts_embed[0] # node text feature
        arxiv_data_list.label_text_feat = texts_embed[1] # label text feature
        arxiv_data_list.edge_text_feat = texts_embed[2] # edge text feature
        arxiv_data_list.prompt_text_feat = texts_embed[3] # prompt node text feature
        arxiv_data_list.prompt_edge_feat = texts_embed[4] # prompt edge text feature

        # get dataset split
        split_idx = arxiv_data.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        arxiv_data_list.train_mask = train_idx
        arxiv_data_list.val_mask = valid_idx
        arxiv_data_list.test_mask = test_idx

        data, slices = self.collate([arxiv_data_list]) # Pass the data_list as a list

        torch.save((data, slices), self.processed_paths[0])
        print("Arxiv is processed. Saved.")
