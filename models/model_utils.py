from torch_scatter import scatter
import torch.nn as nn
import torch
from models.general_gnn import GeneralGNN
import torch.nn.functional as F


# After message passing in subgraph, aggregate the information of target nodes in the supernode
def supernode_aggr(x_feat, supernode_edge_idx, supernode_indices, aggr='mean'):
    # scatter sums up the x_feat of target nodes
    supernode_feat = scatter(src=x_feat[supernode_edge_idx[0]], index=supernode_edge_idx[1], dim=0, reduce=aggr)
    supernode_feat = supernode_feat[supernode_indices, :]


class SingleHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.sqrt = torch.sqrt(torch.tensor(out_dim))

        self.Wk = nn.Parameter(torch.zeros((out_dim, out_dim)))
        nn.init.xavier_uniform_(self.Wk)

        self.Wq = nn.Parameter(torch.zeros((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.Wq)

    def forward(self, k, q, v):
        key = torch.matmul(k, self.Wk) # N x L x l2
        key_T = key.transpose(1,2) # N x l2 x L
        query = torch.matmul(q, self.Wq) # N x l2
        query = query.unsqueeze(1) # N x 1 x l2

        score = torch.matmul(query, key_T) / self.sqrt # N x 1 x L
        attention = F.softmax(score, dim=-1)
        context = torch.matmul(attention, v) # N x 1 x l2

        return context.squeeze() # N x l2

class EdgeTypeMultiLayerMessagePassing(nn.Module):
    def __init__(
        self,
        name,
        num_layers,
        in_dim,
        out_dim,
        num_relations=1,
        edge_attr_dim=None,
        heads=1,
        dropout=0,
        JK=None,
        aggr='mean',
        batch_norm=True,
        **kwargs
    ):
        super().__init__()

        self.name = name
        self.num_layers = num_layers
        self.JK = JK
        self.inp_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_relations = num_relations
        self.heads = heads
        self.dropout = nn.Dropout(p=dropout)
        self.aggr = aggr
        self.batch_norm = batch_norm

        self.module_list = torch.nn.ModuleList()
        self.mlp_list = torch.nn.ModuleList()
        self.batch_norm_list = torch.nn.ModuleList() if batch_norm else None

        self.build_layers()

    def build_layers(self):
        for layer in range(self.num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(self.out_dim, 2*self.out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*self.out_dim, self.out_dim)
            )
            self.mlp_list.append(mlp)

            if self.batch_norm:
                self.batch_norm_list.append(nn.BatchNorm1d(self.out_dim))

            if layer == 0:
                self.module_list.append(self.build_input_layer())
            else:
                self.module_list.append(self.build_hidden_layer())

    def build_input_layer(self):
        return GeneralGNN(
            name=self.name,
            in_dim=self.inp_dim,
            out_dim=self.out_dim,
            edge_attr_dim=self.edge_attr_dim,
            num_relations=self.num_relations,
            heads=self.heads,
            aggr=self.aggr,
        )

    def build_hidden_layer(self):
        return GeneralGNN(
            name=self.name,
            in_dim=self.out_dim,
            out_dim=self.out_dim,
            edge_attr_dim=self.edge_attr_dim,
            num_relations=self.num_relations,
            heads=self.heads,
            aggr=self.aggr,
        )

    def forward(self, x, edge_index, edge_attr, edge_type):
        h_list = []

        h = x
        for layer in range(self.num_layers):
            h = self.module_list[layer].forward(x=h, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
            h = self.mlp_list[layer](h)

            if self.batch_norm:
                h = self.batch_norm_list[layer](h)

            if layer != self.num_layers - 1:
                h = F.relu(h)

            h = self.dropout(h)
            h_list.append(h)

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        else:
            repr = torch.stack(h_list, dim=1)
        return repr


