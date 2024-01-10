
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GNN(nn.Module):
    def __init__(self, name="gcn", num_classes=7):
        super().__init__()

        self.name = name.lower()
        self.dropout = nn.Dropout(p=0.3)
        self.num_classes = num_classes

        if self.name == "gcn":
            self.conv1 = GCNConv(768, 64) # 768 is hardcoed because LLM output dimension is 768
            self.conv2 = GCNConv(64, self.num_classes)

        elif self.name == "gat":
            # 768 is hardcoed because LLM output dimension is 768
            self.conv1 = GATConv(768, 16, heads=4)
            self.conv2 = GATConv(16 * 4, self.num_classes)

        elif self.name == "sage" or self.name == "graphsage":
            # 768 is hardcoed because LLM output dimension is 768
            self.conv1 = SAGEConv(768, 64, normalize=True, project=True)
            self.conv2 = SAGEConv(64, self.num_classes, normalize=True, project=True)

        else:
            raise NotImplementedError

    def forward(self, data):
        x, edge_index = data.x_text_feat, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


