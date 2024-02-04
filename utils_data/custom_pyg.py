from torch.utils.data import Dataset
from torch_geometric.loader import NeighborLoader
import torch

# Empty class type, all datasets inherit from this class
class CustomPygDataset:
    pass

class SubgraphPygDataset(Dataset):
    def __init__(self, graph, num_neighbors=[-1], subgraph_type="induced"):
        self.graph = graph
        self.num_neighbors = num_neighbors
        self.subgraph_type = subgraph_type

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        elif isinstance(index, dict):
            return {key: self.__getitem__(value) for key, value in index.items()}
        elif not isinstance(index, int):
            raise IndexError("Only integers, lists and dictionaries can be used as indices")

        assert index >= 0 and index < len(self)

        loader = NeighborLoader(data=self.graph,
                                num_neighbors=self.num_neighbors,
                                input_nodes=torch.LongTensor([index]),
                                subgraph_type=self.subgraph_type)
        subgraph = next(iter(loader))
        subgraph.batch_size = None

        return subgraph

    def __len__(self):
        return self.graph.num_nodes