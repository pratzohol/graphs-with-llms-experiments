import torch.nn as nn
import torch
from models.model_utils import *


class SuperModel(nn.Module):
    def __init__(self, model_option, model_params):
        super().__init__()

        self.model_option = model_option
        self.model_params = model_params

        if self.model_option == 1:
            self.model1_bg_gnn = None

        elif self.model_option == 2:
            self.model2_gnn = EdgeTypeMultiLayerMessagePassing(**self.model_params)
            self.attn_model = SingleHeadAttention(in_dim=self.model_params["in_dim"], out_dim=self.model_params["out_dim"])
            self.link_pred_mlp = nn.Sequential(
                nn.Linear(self.model_params["out_dim"], 2*self.model_params["out_dim"]),
                nn.ReLU(),
                nn.Linear(2*self.model_params["out_dim"], self.model_params["out_dim"]),
                nn.ReLU(),
                nn.Linear(self.model_params["out_dim"], 1)
            )
        else:
            raise ValueError(f"Unknown model option: {self.model_option}")

    # inputs come from each batch of dataloader
    def forward1(self, **kwargs):
        pass
        # graphs = kwargs["graphs"]


        # supernode_indices = graphs.ptr[:-1] + graphs.supernode
        # target_node_indices = graphs.ptr[:-1]

        # x_feat_new = self.multilayer_gnn.forward()

        # supernode_feat = supernode_aggr(x_feat_new, graphs.edge_index_supernode, supernode_indices)

        # x_target_node, x_label = self.attentional_gnn.foward()

    def forward2(self, **kwargs):
        graphs = kwargs["graphs"]
        label_indices = kwargs["label_indices"]
        correct_label_mask = kwargs["correct_label_mask"]

        x = graphs.x_text_feat
        e_idx = graphs.edge_index
        e_attr = graphs.edge_attr
        e_type = graphs.edge_type

        h = self.model2_gnn.forward(x=x, edge_index=e_idx, edge_attr=e_attr, edge_type=e_type)
        h = self.attn_model.forward(k=h, q=x, v=h)

        class_embeddings = h[label_indices]
        return self.link_pred_mlp.forward(class_embeddings), correct_label_mask

    def forward(self, **kwargs):
        if self.model_option == 1:
            return self.forward1(**kwargs)
        elif self.model_option == 2:
            return self.forward2(**kwargs)
        else:
            raise ValueError(f"Unknown model option: {self.model_option}")
