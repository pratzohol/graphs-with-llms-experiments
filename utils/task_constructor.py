import numpy as np
from utils_data.custom_pyg import CustomPygDataset

class TaskConstructor:
    def __init__(self, custom_data : CustomPygDataset, split="train_mask"):
        self.custom_data = custom_data
        self.graph_data = custom_data._data
        self.split = split

        self.num_classes = self.custom_data.num_classes
        self.unique_labels_set = range(self.num_classes)  # list of unique labels

        all_labels = self.graph_data.y.squeeze().numpy()  # returns numpy array on cpu
        train_split_idx = self.graph_data["train_mask"]

        all_labels = -1 - all_labels  # flip labels
        all_labels[train_split_idx] = -1 - all_labels[train_split_idx] # flip labels of train back, others will be -ve

        self.train_activated_labels = all_labels
        self.train_label2idx = {label: np.where(self.train_activated_labels == label)[0] for label in self.unique_labels_set} # for that particular label, get the indices of train

        if split != "train_mask":
            all_labels = self.graph_data.y.squeeze().numpy()
            split_idx = self.graph_data[split]

            all_labels = -1 - all_labels  # flip labels
            all_labels[split_idx] = -1 - all_labels[split_idx] # flip labels of val/test back, others will be -ve

            self.split_activated_labels = all_labels
            self.label2idx = {label: np.where(self.split_activated_labels == label)[0] for label in self.unique_labels_set} # for that particular label, get the indices of train/val/test


    def sample(self, params):
        rng = rng
        sampled_labels = rng.sample(self.unique_labels_set, params["n_way"])

        task = {}
        if self.split == "train_mask":
            for label in sampled_labels:
                members = self.train_label2idx[label] # numpy array
                sample_fn = rng.choices if members.shape[0] < params["n_member"] else rng.sample # choices = replacement, sample = no replacement and unique
                task[label] = members[sample_fn(range(members.shape[0]), k=params["n_member"])].tolist() # convert to list
        else:
            for label in sampled_labels:
                members = self.label2idx[label]
                train_members = self.train_label2idx[label]

                sample_fn = rng.choices if members.shape[0] < params["n_query"] else rng.sample
                trn_sample_fn = rng.choices if train_members.shape[0] < params["n_shot"] else rng.sample

                few_shot =  train_members[trn_sample_fn(range(train_members.shape[0]), k=params["n_shot"])].tolist()
                queries = members[sample_fn(range(members.shape[0]), k=params["n_query"])].tolist()

                task[label] = few_shot + queries

        return task
