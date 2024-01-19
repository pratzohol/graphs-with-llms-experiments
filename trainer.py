import pandas as pd
import numpy as np
import torch
from models.gnn import GNN
from models.mlp import MLP
from utils.dataloader import GetDataloader
from tqdm import trange
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp


class Trainer:
    def __init__(self, dataset_name="cora", sentence_encoder="ST", model_type="mlp", device=0, state_dict_path="./state_dicts"):
        self.dataset_name = dataset_name
        self.sentence_encoder = sentence_encoder
        self.model_type = model_type.lower()
        self.device = torch.device("cpu" if device==123 else f"cuda:{device}")

        self.state_dict_path = osp.join(state_dict_path, f"{self.dataset_name}_{self.sentence_encoder}", f"{model_type}")
        if not osp.exists(self.state_dict_path):
            os.makedirs(self.state_dict_path)

        dataloader = GetDataloader(dataset_name=self.dataset_name, sentence_encoder=self.sentence_encoder, device=self.device)
        self.data = dataloader.get_data()
        self.num_classes = len(self.data.y.squeeze().unique())

        if self.model_type == "mlp":
            self.model = MLP(num_classes=self.num_classes)
        elif self.model_type in ["gcn", "gat", "sage", "graphsage"]:
            self.model = GNN(name=self.model_type, num_classes=self.num_classes)
        else:
            raise NotImplementedError

        self.data = self.data.to(device=self.device)
        self.model = self.model.to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)

    def train(self, mask_idx):
        best_val_acc = 0

        # total of 10 training masks are present for each dataset
        for e in range(1, 201):
            self.model.train()
            self.optimizer.zero_grad()

            out = self.model(self.data)
            train_pred = out.argmax(dim=1)

            train_ypred = train_pred[self.data.train_masks[mask_idx]]
            train_ytrue = self.data.y[self.data.train_masks[mask_idx]]

            train_correct = (train_ypred == train_ytrue).sum()

            train_acc = int(train_correct) / train_ytrue.shape[0]
            train_loss = F.cross_entropy(out[self.data.train_masks[mask_idx]], train_ytrue)

            if e % 10 == 0:
                val_ypred = train_pred[self.data.val_masks[mask_idx]]
                val_ytrue = self.data.y[self.data.val_masks[mask_idx]]
                val_correct = (val_ypred == val_ytrue).sum()

                val_acc = int(val_correct) / val_ytrue.shape[0]
                val_loss = F.cross_entropy(out[self.data.val_masks[mask_idx]], val_ytrue)

                print(f"Epoch {e} => Train Accuracy : {train_acc} | Train Loss : {train_loss}")
                print(f"Validation Accuracy : {val_acc} | Validation Loss : {val_loss}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                    save_path = osp.join(self.state_dict_path, f"Mask_{mask_idx}_best_state_dict.pt")
                    if osp.exists(save_path):
                        os.remove(save_path)

                    model_info = {"state_dict" : self.model.state_dict(),
                                    "optimizer_state_dict" : self.optimizer.state_dict(),
                                    "val_accuracy" : best_val_acc,
                                    "val_loss" : val_loss}

                    torch.save(model_info, save_path)

            train_loss.backward()
            self.optimizer.step()


        self.model.eval()
        with torch.inference_mode():
            out = self.model(self.data)
            pred = out.argmax(dim=1)

            ypred = pred[self.data.test_masks[mask_idx]]
            ytrue = self.data.y[self.data.test_masks[mask_idx]]
            test_correct = (ypred == ytrue).sum()

            test_acc = int(test_correct) / ytrue.shape[0]
            test_loss = float(F.cross_entropy(out[self.data.test_masks[mask_idx]], ytrue))

        return test_acc, test_loss