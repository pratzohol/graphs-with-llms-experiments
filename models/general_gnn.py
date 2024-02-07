import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATv2Conv, GINEConv, RGATConv
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, add_self_loops

# General GNN to handle edge_attributes
class GeneralGNN(nn.Module):
    def __init__(self, name="sage", in_dim=768, out_dim=256, edge_attr_dim=None, num_relations=1, heads=1, aggr='mean'):
        super().__init__()

        self.name = name.lower()
        # These 3 can handle edge_attributes only
        if self.name in ["sage", "graphsage"]:
            self.conv_ = SAGEConv(in_dim, out_dim, edge_attr_dim, aggr=aggr)
        elif self.name == "gin":
            nn_ = nn.Sequential(
                nn.Linear(in_dim, 2*out_dim),
                nn.ReLU(),
                nn.Linear(2*out_dim, out_dim)
            )
            self.conv_ = GINEConv(
                nn=nn_,
                edge_dim=edge_attr_dim,
                aggr=aggr
            )
        elif self.name == "gat":
            self.conv_ = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads,
                edge_dim=edge_attr_dim,
                aggr=aggr
            )

        # These 2 can handle edge_types and edge_attributes both
        elif self.name == "rgcn":
            self.conv_ = RGCNConv(in_dim, out_dim, edge_attr_dim, num_relations, aggr)
        elif self.name == "rgat":
            self.conv = RGATConv(
                in_channels=in_dim,
                out_channels=out_dim,
                num_relations=num_relations,
                heads=heads,
                edge_dim=edge_attr_dim,
                aggr=aggr
            )

            self.conv_ = nn.Sequential(self.conv, nn.Linear(out_dim*heads, out_dim))
        else:
            raise ValueError(f"Unknown GNN type: {self.name}")

    def forward(self, x, edge_index, edge_attr, edge_type=None):
        if edge_type is None:
            return self.conv_.forward(x, edge_index, edge_attr)
        else:
            return self.conv_.forward(x, edge_index, edge_attr, edge_type)


class SAGEConv(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_attr_dim, aggr="mean"):
        super().__init__()

        self.lin_x = torch.nn.Linear(in_dim, out_dim)
        self.lin_self_loops = torch.nn.Linear(in_dim, out_dim)
        self.lin_edge_attr = None if edge_attr_dim is None else torch.nn.Linear(edge_attr_dim, out_dim)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr=None):
        x_transformed = self.lin_x(x)

        if self.lin_edge_attr is None:
            x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed)
        else:
            edge_attr_emb = self.lin_edge_attr(edge_attr)
            x_msg = self.propagate(aggr=self.aggr, edge_index=edge_index, x=x_transformed, edge_attr=edge_attr_emb)

        x_msg += self.lin_self_loops(x)
        return x_msg

    def message(self, x_j, edge_attr=None):
        return x_j if edge_attr is None else x_j + edge_attr


class RGCNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_attr_dim=None, num_relations=1, aggr='mean'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.aggr = aggr

        self.weight = Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.weight_self = Parameter(torch.empty(in_dim, out_dim))
        self.bias = Parameter(torch.empty(out_dim))

        self.reset_parameters()

        self.lin_edge_attr = None if edge_attr_dim is None else torch.nn.Linear(edge_attr_dim, in_dim)

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.weight_self)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        if self.lin_edge_attr is not None:
            edge_attr_emb = self.lin_edge_attr(edge_attr)

        out = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        for i in range(self.num_relations):
            edge_mask = edge_type == i
            e_idx = edge_index[:, edge_mask]

            if self.lin_edge_attr is None:
                h = self.propagate(aggr=self.aggr, edge_index=e_idx, x=x)
            else:
                e_attr_emb = edge_attr_emb[edge_mask]
                h = self.propagate(aggr=self.aggr, edge_index=e_idx, x=x, edge_attr=e_attr_emb)

            out += h @ self.weight[i]

        out += x @ self.weight_self
        out += self.bias

        return out

    def message(self, x_j, edge_attr=None):
        return x_j if edge_attr is None else x_j + edge_attr
