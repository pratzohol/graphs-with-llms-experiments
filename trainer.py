import pandas as pd
import numpy as np
import torch
from models.gnn import GNN
from models.mlp import MLP
from utils.dataloader import GetDataloader
from tqdm import trange, trange
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import wandb
import time


class Trainer:
    def __init__(self, params):
        wandb.init(project="graphs-with-llms-experiments", name=params["exp name"])
        wandb.run.summary["wandb_url"] = wandb.run.url

        self.params = params

        self.dataRoot = params["dataRoot"]
        self.custom_dataRoot = params["custom_dataRoot"]

        self.dataset_name = params["dataset"]
        self.sentence_encoder = params["sentence_encoder"]
        self.model_type = params["model_type"].lower()
        self.device = params["device"]
        self.epochs = params["epochs"]
        self.batch_ocunt = params["batch_count"]


        self.state_dict_path = osp.join(params["state_dict_path"], f"{self.dataset_name}_{self.sentence_encoder}", f"{self.model_type}")
        if not osp.exists(self.state_dict_path):
            os.makedirs(self.state_dict_path)

        dataloader = GetDataloader(**self.params)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader.get_dataloader()
        self.num_classes = dataloader.get_num_classes()
        # self.data = dataloader.get_data()

        if self.model_type == "mlp":
            self.model = MLP(num_classes=self.num_classes, dropout=params["dropout"])
        elif self.model_type in ["gcn", "gat", "sage", "graphsage"]:
            self.model = GNN(name=self.model_type, num_classes=self.num_classes, dropout=params["dropout"])
        else:
            raise NotImplementedError

        # self.data = self.data.to(device=self.device)
        self.model = self.model.to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        # total number of model parameters
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = int(sum([np.prod(p.size()) for p in model_parameters]))
        print("Number of trainable parameters of the model:", num_params)
        wandb.run.summary["num_params"] = num_params

        wandb.config.params = params
        wandb.watch(self.model, log_freq=100)


    def train(self):
        total_time = 0
        best_val_acc = 0

        trn_dtldr_itr = iter(self.train_dataloader)

        self.model.eval()
        with torch.inference_mode():
            # TODO: Change the logic for test and how testing is done


            wandb.log({"test_acc": test_acc, "test_loss": test_loss})

        if self.params["eval_only"]:
            print("Evaluation only - skipping training - exiting now")
            print("Note: also skipping evaluation of val set")
            return test_acc, test_loss

        for e in range(self.epochs):
            tbar = trange(1, self.batch_count)
            for step in tbar:
                #####################################################
                t1 = time.time()
                #####################################################

                try:
                    batch = next(trn_dtldr_itr)
                except StopIteration:
                    trn_dtldr_itr = iter(self.train_dataloader)
                    batch = next(trn_dtldr_itr)

                #####################################################
                t2 = time.time()
                #####################################################

                # TODO: Change self.data as we now use batch, transfer batch to self.device and change the training logic
                self.model.train()
                self.optimizer.zero_grad()

                batch = [i.to(self.device) for i in batch] # move to gpu device

                out = self.model(*batch)
                # train_pred = out.argmax(dim=1)

                # train_ypred = train_pred[batch.train_mask]
                # train_ytrue = batch.y[batch.train_mask]

                # train_correct = (train_ypred == train_ytrue).sum()

                # train_acc = int(train_correct) / train_ytrue.shape[0]
                # train_loss = F.cross_entropy(out[batch.train_mask], train_ytrue)

                train_loss.backward()
                self.optimizer.step()

                #####################################################
                t3 = time.time()
                #####################################################

                wandb.log({"batch_training_time": t3 - t2}, step=e)
                wandb.log({"batch_loading_time": t2 - t1}, step=e) # avg time to load and process a single batch.
                wandb.log({"train_acc": train_acc, "train_loss": train_loss}, step=e)

                total_time += t3 - t1
                tbar.set_description(f"Epoch: {e} | Step: {step} / {self.batch_count}")

                # TODO: Change the logic for validation and when to run and how to store best val
                if step % 1000 == 0:


                    wandb.log({"val_acc": val_acc, "val_loss": val_loss}, step=e)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                        save_path = osp.join(self.state_dict_path, f"best_state_dict.pt")
                        if osp.exists(save_path):
                            os.remove(save_path)

                        model_info = {"state_dict" : self.model.state_dict(),
                                        "optimizer_state_dict" : self.optimizer.state_dict(),
                                        "val_accuracy" : best_val_acc,
                                        "val_loss" : val_loss}

                        torch.save(model_info, save_path)

        return test_acc, test_loss