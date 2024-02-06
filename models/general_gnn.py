import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATv2Conv, GINEConv, RGATConv, RGCNConv
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
            self.conv_ = RGCNConv(
                in_channels=in_dim,
                out_channels=out_dim,
                num_relations=num_relations,
                edge_dim=edge_attr_dim,
                aggr=aggr
            )
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
