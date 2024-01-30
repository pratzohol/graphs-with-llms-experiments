from ogb.nodeproppred import PygNodePropPredDataset
import os
import os.path as osp
import torch_geometric
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url

from utils.encoder import SentenceEncoder

# Loading the cora dataset
data_root = "../data"

class CoraPyGDataset(InMemoryDataset):
    def __init__(self, dataRoot="data", custom_dataRoot="custom_data", sentence_encoder=None, transform=None, pre_transform=None, pre_filter=None):
        self.data_root = dataRoot
        self.custom_data_root = custom_dataRoot
        self.sentence_encoder = sentence_encoder
        self.custom_data_dir = osp.join(self.custom_data_root, f"cora_{self.sentence_encoder.name}")

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
            return self.sentence_encoder.encode(texts)  # returns to self.device

    def encode_texts(self, texts):
        if isinstance(texts[0], str):
            return self.text_to_embed(texts)
        return [self.text_to_embed(t) for t in texts]

    def generate_custom_data(self):
        # Load the raw cora dataset
        data_path = osp.join(self.data_root, "cora", "cora.pt")
        raw_cora_data = torch.load(data_path)

        texts = raw_cora_data.raw_text
        label_names = raw_cora_data.label_names


        # Label and label description
        category_desc = pd.read_csv(osp.join(self.data_root, "cora", "categories.csv"), sep=",").values

        # Sort the label desc by the order of label_names
        ordered_desc = []
        for i, label in enumerate(label_names):
            true_ind = (label == category_desc[:, 0])
            ordered_desc.append((label, category_desc[true_ind, 1][0]))

        # Prompts for nodes/edges in original graph (can be changed accordingly)
        node_texts = ["Feature Node.\n Paper Title and abstract: " + t for t in texts]
        edge_text = ["Feature Edge.\n Connected papers are cited together by other papers."]

        # Node classification : Prompts for prompt node and label node (can be changed accordingly)
        prompt_node_text = ["Prompt Node.\n Node Classification on the paper's category"]
        label_texts = ["Prompt Node.\n Literature Category and Description: " + desc[0] + " + " + desc[1] for desc in ordered_desc]

        # Link prediction : Prompts for prompt node and edge labels (can be changed accordingly)
        prompt_node_edge_text = ["Prompt Node.\n Link Prediction on the papers that are cited together"]
        edge_label_text = ["Prompt Node.\n Two papers have co-citation",
                           "Prompt Node.\n Two papers do not have co-citation"]

        # Prompt for edge b/w prompt node and labels (can be changed accordingly)
        prompt_edge_text = ["Prompt Edge."]

        return raw_cora_data, [node_texts, label_texts, edge_text, prompt_node_edge_text, prompt_node_text, prompt_edge_text, edge_label_text]

    def process(self):
        # raw cora dataset is not in any library, so we process and load it manually in self.generate_custom_data()
        cora_data_list, texts = self.generate_custom_data()
        texts_embed = self.encode_texts(texts)

        torch.save(texts, self.processed_paths[1])

        cora_data_list.x_text_feat = texts_embed[0]
        cora_data_list.label_text_feat = texts_embed[1]
        cora_data_list.edge_text_feat = texts_embed[2]
        cora_data_list.prompt_text_edge_feat = texts_embed[3]
        cora_data_list.prompt_text_feat = texts_embed[4]
        cora_data_list.prompt_edge_feat = texts_embed[5]
        cora_data_list.edge_label_feat = texts_embed[6]

        # Initially 'cora.pt' has 10 different masks. We use only the first one.
        cora_data_list.train_mask = cora_data_list.train_masks[0]
        cora_data_list.val_mask = cora_data_list.val_masks[0]
        cora_data_list.test_mask = cora_data_list.test_masks[0]

        cora_data_list.train_masks = None
        cora_data_list.val_masks = None
        cora_data_list.test_masks = None

        # Pass the data_list as a list
        data, slices = self.collate([cora_data_list])

        torch.save((data, slices), self.processed_paths[0])
        print("Cora is processed. Saved.")
