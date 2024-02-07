import numpy as np
import torch
from utils_data.custom_pyg import CustomPygDataset

class TaskConstructor:
    def __init__(self, custom_data : CustomPygDataset, split="train_mask"):
        self.custom_data = custom_data
        self.graph_data = custom_data._data.to('cpu')
        self.split = split

        self.mask = self.graph_data[split]
        self.mask = torch.nonzero(self.mask).squeeze() if self.mask.dtype == torch.bool else self.mask
        self.mask_len = len(self.mask)

        self.num_classes = self.custom_data.num_classes
        self.unique_labels_set = range(self.num_classes)  # list of unique labels

        all_labels = self.graph_data.y.numpy()  # returns numpy array on cpu
        train_split_idx = self.graph_data["train_mask"]

        all_labels = -1 - all_labels  # flip labels
        all_labels[train_split_idx] = -1 - all_labels[train_split_idx] # flip labels of train back, others will be -ve

        self.train_activated_labels = all_labels
        self.train_label2idx = {label: np.where(self.train_activated_labels == label)[0] for label in self.unique_labels_set} # for that particular label, get the indices of train

        if split != "train_mask":
            all_labels = self.graph_data.y.numpy()
            split_idx = self.graph_data[split]

            all_labels = -1 - all_labels  # flip labels
            all_labels[split_idx] = -1 - all_labels[split_idx] # flip labels of val/test back, others will be -ve

            self.split_activated_labels = all_labels
            self.label2idx = {label: np.where(self.split_activated_labels == label)[0] for label in self.unique_labels_set} # for that particular label, get the indices of train/val/test


    def sample(self, params, eval_idx=0):
        rng = params["rng"]
        task = {}

        if self.split == "train_mask":
            sampled_labels = rng.sample(self.unique_labels_set, params["n_way"])
            for label in sampled_labels:
                members = self.train_label2idx[label] # numpy array
                sample_fn = rng.choices if members.shape[0] < params["n_shot"] else rng.sample # choices = replacement, sample = no replacement and unique
                task[label] = members[sample_fn(range(members.shape[0]), k=params["n_shot"])].tolist() # convert to list
        else:
            eval_node = self.mask.tolist()[eval_idx]
            eval_idx_label = self.graph_data.y.numpy()[eval_node]

            modified_all_labels = [l for l in self.unique_labels_set if l != eval_idx_label]
            sampled_labels = rng.sample(modified_all_labels, params["n_way"] - 1)
            for label in sampled_labels:
                train_members = self.train_label2idx[label]
                trn_sample_fn = rng.choices if train_members.shape[0] < params["n_shot"] else rng.sample
                task[label] = train_members[trn_sample_fn(range(train_members.shape[0]), k=params["n_shot"])].tolist()

            members = self.train_label2idx[eval_idx_label]
            sample_fn = rng.choices if members.shape[0] < params["n_shot"] else rng.sample
            task[eval_idx_label] = members[sample_fn(range(members.shape[0]), k=params["n_shot"] - 1)].tolist()
            task[eval_idx_label].append(eval_node)

        return task
