from typing import Any
import torch
import os
import os.path as osp
import sys
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from itertools import chain

sys.path.append(osp.join(osp.dirname(__file__), ".."))

from utils_data.cora import CoraPyGDataset
from utils_data.pubmed import PubmedPyGDataset
from utils_data.ogbn_arxiv import ArxivPyGDataset
from utils_data.ogbn_products import ProductsPyGDataset
from utils_data.custom_pyg import SubgraphPygDataset
from utils.encoder import SentenceEncoder
from utils.task_constructor import TaskConstructor
from utils.sampler import TrainBatchSampler, EvalBatchSampler


class GetDataloader:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs["dataset"].lower()
        self.dataRoot = kwargs["dataRoot"]
        self.custom_dataRoot = kwargs["custom_dataRoot"]
        self.device = kwargs["device"]

        self.sentence_encoder = SentenceEncoder(name=kwargs["sentence_encoder"], root=kwargs["encoder_path"], device=self.device)

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

        self.graph = self.get_data().to('cpu')

        smplr_params = {
            "batch_count": kwargs["batch_count"],
            "batch_size": kwargs["batch_size"],
            "seed": kwargs["seed"],
            "n_way": kwargs["n_way"],
            "n_shot": kwargs["n_shot"],
            "n_query": kwargs["n_query"],
            "n_member" : kwargs["n_shot"] + kwargs["n_query"],
            "leave_last": False,
        }

        self.trn_smplr = TrainBatchSampler(params=smplr_params, task=TaskConstructor(self.custom_data, split="train_mask"))
        self.val_smplr = EvalBatchSampler(params=smplr_params, task=TaskConstructor(self.custom_data, split="val_mask"))
        self.test_smplr = EvalBatchSampler(params=smplr_params, task=TaskConstructor(self.custom_data, split="test_mask"))

        self.subgraph_custom_data = SubgraphPygDataset(graph=self.graph, num_neighbors=kwargs["num_neighbors"], subgraph_type=kwargs["subgraph_type"])

        model_option = kwargs["model_option"]
        if model_option == 1:
            collate = Collator(params=smplr_params)
        elif model_option == 2:
            collate = Collator2(params=smplr_params)
        else:
            raise ValueError(f"Unknown model option: {model_option}")

        self.train_dataloader = DataLoader(self.subgraph_custom_data, batch_sampler=self.trn_smplr, collate_fn=collate)
        self.val_dataloader = DataLoader(self.subgraph_custom_data, batch_sampler=self.val_smplr, collate_fn=collate)
        self.test_dataloader = DataLoader(self.subgraph_custom_data, batch_sampler=self.test_smplr, collate_fn=collate)


    def get_dataloader(self):
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def get_num_classes(self):
        return self.custom_data.num_classes

    def get_data(self):
        return self.custom_data._data


class Collator:
    def __init__(self, params):
        self.n_shot = params["n_shot"]
        self.n_query = params["n_query"]

    def process_one_graph(self, graph):
        node_attrs = [key for key, value in graph if graph.is_node_attr(key)]
        for key in node_attrs:
            value = graph[key]
            if isinstance(value, torch.Tensor):
                graph[key] = torch.cat((value, torch.zeros(1, *value.shape[1:], dtype=value.dtype, layout=value.layout, device=value.device)))

        supernode_idx = graph.num_nodes
        graph.supernode = torch.tensor([supernode_idx])
        graph.edge_index_supernode = torch.tensor([[0], [supernode_idx]], dtype=torch.long)
        graph.num_nodes += 1

        return graph

    def process_one_task(self, task):
        label_map = list(task)
        label_map_reverse = {v: i for i, v in enumerate(label_map)}

        all_graphs = []
        labels = []
        query_mask = []

        for label, graphs in task.items():
            all_graphs.extend([self.process_one_graph(g) for g in graphs])
            query_mask.extend([False] * self.n_shot)
            query_mask.extend([True] * self.n_query)
            labels.extend([label_map_reverse[label]] * len(graphs)) # label_map_reverse[label] is the index of label in label_map

        return all_graphs, torch.tensor(labels), torch.tensor(query_mask), label_map

    def __call__(self, batch):
        graphs, labels, query_mask, label_map = map(list, zip(*[self.process_one_task(task) for task in batch]))

        num_task = len(graphs)
        task_len = len(graphs[0])
        num_labels = len(label_map[0])

        graphs_ = Batch.from_data_list([g for l in graphs for g in l])
        labels_ = torch.cat(labels)
        query_mask_ = torch.cat(query_mask)
        label_map_ = list(chain(*label_map))

        metagraph_edge_source = torch.arange(labels_.size(0)).repeat_interleave(num_labels)

        metagraph_edge_target = torch.arange(num_labels).repeat(labels_.size(0))
        metagraph_edge_target += (torch.arange(num_task) * num_labels).repeat_interleave(task_len * num_labels) + labels_.size(0)

        metagraph_edge_index = torch.stack([metagraph_edge_source, metagraph_edge_target], dim=0)

        metagraph_edge_mask = query_mask_.repeat_interleave(num_labels) # True for query_mask

        metagraph_edge_attr = torch.nn.functional.one_hot(labels_, num_labels).float()
        metagraph_edge_attr = metagraph_edge_attr.reshape(-1)
        metagraph_edge_attr = (metagraph_edge_attr * 2 - 1) * (~metagraph_edge_mask)

        metagraph_edge_attr = torch.stack([metagraph_edge_mask, metagraph_edge_attr], dim=1)

        all_label_embeddings = graphs[0][0].label_text_feat

        label_map_ = torch.tensor(label_map_)
        label_embeddings = all_label_embeddings[label_map_]

        labels_onehot = torch.nn.functional.one_hot(labels_).float()

        final_batch = {
            "graphs": graphs_,
            "label_embeddings": label_embeddings,
            "labels_onehot": labels_onehot,
            "metagraph_edge_index": metagraph_edge_index,
            "metagraph_edge_attr": metagraph_edge_attr,
            "metagraph_edge_mask": metagraph_edge_mask
        }
        return final_batch

class Collator2:
    def __init__(self, params):
        self.params = params
        self.n_way = params["n_way"]
        self.n_shot = params["n_shot"]

    def process_one_graph(self, graph):
        node_attrs = [key for key, value in graph if graph.is_node_attr(key)]
        for key in node_attrs:
            value = graph[key]
            if isinstance(value, torch.Tensor):
                graph[key] = torch.cat((value, torch.zeros(1, *value.shape[1:], dtype=value.dtype, layout=value.layout, device=value.device)))

        supernode_idx = graph.num_nodes
        graph.supernode = torch.tensor([supernode_idx])
        graph.x_text_feat[graph.supernode] = graph.prompt_text_feat[0]

        num_edges = graph.edge_index.shape[1]

        # Edge types -> 0 : original edges, 1 : edges to supernode, 2 : edges from supernode
        prompt_edge = graph.prompt_edge_feat
        if num_edges > 0:
            edge_attr = torch.stack([graph.edge_text_feat[0]] * num_edges)
            graph.edge_attr = torch.cat([edge_attr, prompt_edge, prompt_edge])

            edge_type = torch.zeros(num_edges, dtype=torch.long)
            graph.edge_type = torch.cat([edge_type, torch.tensor([1, 2], dtype=torch.long)])
        else:
            graph.edge_attr = torch.cat([prompt_edge, prompt_edge])
            graph.edge_type = torch.tensor([1, 2], dtype=torch.long)

        edge_index = torch.tensor([[0, supernode_idx], [supernode_idx, 0]], dtype=torch.long)
        graph.edge_index = torch.cat([graph.edge_index, edge_index], dim=1)

        if graph.edge_attr.shape[0] != graph.edge_index.shape[1]:
            print(graph.edge_attr.shape, graph.edge_index.shape, graph.edge_type.shape)
            print(num_edges)

        assert graph.edge_index.shape[1] == graph.edge_attr.shape[0]
        assert graph.edge_index.shape[1] == graph.edge_type.shape[0]

        graph.num_nodes += 1
        return graph

    def process_one_task(self, task):
        label_map = torch.tensor(list(task))
        query_label = label_map[-1]

        correct_label_mask = torch.tensor([False] * len(label_map))
        correct_label_mask[-1] = True

        indices_shuffled = torch.randperm(len(label_map))
        label_map = label_map[indices_shuffled]
        correct_label_mask = correct_label_mask[indices_shuffled]

        all_graphs = []
        labels = []
        query_mask = []

        for label in label_map.tolist():
            graphs = task[label]

            all_graphs.extend([self.process_one_graph(g) for g in graphs])
            labels.extend([label] * len(graphs))

            query_mask.extend([False] * len(graphs))
            if label == query_label:
                query_mask[-1] = True

        return all_graphs, torch.tensor(labels), torch.tensor(query_mask), label_map, correct_label_mask

    def __call__(self, batch):
        graph_list, labels, query_mask, label_map, correct_label_mask = map(list, zip(*[self.process_one_task(task) for task in batch]))

        num_task = len(graph_list)
        task_len = len(graph_list[0])
        num_labels = len(label_map[0])

        graphs = Batch.from_data_list([g for l in graph_list for g in l])
        query_mask_ = torch.cat(query_mask)
        label_map_ = list(chain(*label_map))
        label_map_ = torch.tensor(label_map_)
        correct_label_mask_ = torch.cat(correct_label_mask)

        all_label_embeddings = graph_list[0][0].label_text_feat
        label_embeddings = all_label_embeddings[label_map_]

        label_indices = torch.arange(num_labels * num_task) + graphs.num_nodes
        supernode_indices = graphs.ptr[:-1] + graphs.supernode

        # Edge Types -> 3: edges from sample to class, 4: edges from query to label, 5: edges from label to query
        smple2cls_edge_source = supernode_indices[~query_mask_]
        smpl2cls_edge_target = label_indices.repeat_interleave(self.n_shot)

        edge_idx = torch.stack([smple2cls_edge_source, smpl2cls_edge_target])
        edge_type = torch.tensor([3] * (edge_idx.shape[1]), dtype=torch.long)

        qry2lbl_edge_source = supernode_indices[query_mask_].repeat_interleave(num_labels)
        qry2lbl_edge_target = label_indices

        qry2lbl_edge = torch.stack([qry2lbl_edge_source, qry2lbl_edge_target])
        qry2lbl_edge_type = torch.tensor([4] * (qry2lbl_edge.shape[1]), dtype=torch.long)

        lbl2qry_edge = qry2lbl_edge.flip(0)
        lbl2qry_edge_type = torch.tensor([5] * (lbl2qry_edge.shape[1]), dtype=torch.long)

        edge_idx = torch.cat([edge_idx, qry2lbl_edge, lbl2qry_edge], dim=1)
        edge_type = torch.cat([edge_type, qry2lbl_edge_type, lbl2qry_edge_type])

        assert edge_idx.shape[1] == edge_type.shape[0]

        prompt_edge_feat = graph_list[0][0].prompt_edge_feat[0]
        edge_attr = torch.stack([prompt_edge_feat] * edge_idx.shape[1])

        graphs.x_text_feat  = torch.cat([graphs.x_text_feat, label_embeddings])

        graphs.edge_index = torch.cat([graphs.edge_index, edge_idx], dim=1)
        graphs.edge_type = torch.cat([graphs.edge_type, edge_type])
        graphs.edge_attr = torch.cat([graphs.edge_attr, edge_attr])

        graphs.num_nodes += num_labels * num_task

        final_batch = {
            "graphs": graphs,
            "label_indices": label_indices,
            "correct_label_mask": correct_label_mask_
        }
        return final_batch

