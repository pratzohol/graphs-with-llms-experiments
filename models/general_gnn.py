import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GeneralGNN(nn.Module):
    def __init__(self, name="gcn", num_classes=7, dropout=0.3):
        super().__init__()

        pass
