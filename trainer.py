import pandas as pd
import numpy as np
import torch
from models.gnn import GNN
from models.mlp import MLP
from utils.loss_acc import *
from utils.dataloader import GetDataloader
from tqdm import trange, trange
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import wandb
import time
from models.model import SuperModel


class Trainer:
    def __init__(self, params):
        wandb.init(project="graphs-with-llms-experiments", name=params["exp name"])
        wandb.run.summary["wandb_url"] = wandb.run.url

        self.params = params

        self.dataRoot = params["dataRoot"]
        self.custom_dataRoot = params["custom_dataRoot"]

        self.dataset_name = params["dataset"]
        self.sentence_encoder = params["sentence_encoder"]
        self.model_option = params["model_option"]
        self.model_params = params["model_params"][self.model_option - 1]
        self.device = params["device"]
        self.epochs = params["epochs"]
        self.batch_count = params["batch_count"]


        self.state_dict_path = osp.join(params["state_dict_path"], f"{self.dataset_name}_{self.sentence_encoder}", f"model_{self.model_option}")
        if not osp.exists(self.state_dict_path):
            os.makedirs(self.state_dict_path)

        dataloader = GetDataloader(**self.params)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader.get_dataloader()
        self.num_classes = dataloader.get_num_classes()

        self.model = SuperModel(self.model_option, self.model_params)
        self.model = self.model.to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        self.loss_fn, self.metric_fn = GetLossAcc(self.model_option, self.params).get_functions()

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

                self.model.train()
                self.optimizer.zero_grad()

                for key in batch:
                    batch[key] = batch[key].to(device=self.device) # move to gpu device

                out = self.model(**batch)

                train_loss = self.loss_fn(out)
                train_acc = self.metric_fn(out)

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

                if step % 1000 == 0:
                    self.model.eval()
                    with torch.inference_mode():
                        val_acc, val_loss = self.evaluate(self.val_dataloader)

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

        self.model.eval()
        with torch.inference_mode():
            test_acc, test_loss = self.evaluate(self.test_dataloader)

        wandb.log({"test_acc": test_acc, "test_loss": test_loss})
        return test_acc, test_loss

    def evaluate(self, dataloader):
        pass