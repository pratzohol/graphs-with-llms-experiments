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
from tqdm import tqdm
import time
from datetime import timedelta
from models.model import SuperModel


def convert_time(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return '%02d:%02d:%02d' % (hour, min, sec)

class Trainer:
    def __init__(self, params):
        wandb.init(project="graphs-with-llms-experiments", notes=params["exp_notes"], name=params["exp_name"])
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
        self.eval_only = params["eval_only"]

        self.state_dict_path = osp.join(params["state_dict_path"], f"{self.dataset_name}_{self.sentence_encoder}", f"model_{self.model_option}")
        if not osp.exists(self.state_dict_path):
            os.makedirs(self.state_dict_path)

        dataloader = GetDataloader(**self.params)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader.get_dataloader()
        self.num_classes = dataloader.get_num_classes()
        self.data = dataloader.get_data().to('cpu')

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
        wandb.watch(self.model, log_freq=10)


    def train(self):
        total_time = 0
        best_val_acc = 0

        trn_dtldr_itr = iter(self.train_dataloader)

        if self.eval_only:
            self.model.eval()
            with torch.inference_mode():
                test_acc, test_loss = self.evaluate(self.test_dataloader, "test")

            print("Final Test accuracy is", test_acc)
            print("Evaluation Mode ONLY. Exiting ...")
            print("Finished")

            wandb.run.summary["final_test_acc"] = test_acc
            wandb.run.summary["final_test_loss"] = test_loss
            wandb.finish()
            return test_acc, test_loss

        tbar = trange(self.epochs * self.batch_count)
        for e in tbar:
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

            train_loss = self.loss_fn(*out)
            correct, total = self.metric_fn(*out)
            train_acc = correct / total

            train_loss.backward()
            self.optimizer.step()

            #####################################################
            t3 = time.time()
            #####################################################

            wandb.log({"batch_training_time": t3 - t2}, step=e)
            wandb.log({"batch_loading_time": t2 - t1}, step=e) # avg time to load and process a single batch.
            wandb.log({"train_acc": train_acc, "train_loss": train_loss}, step=e)

            total_time += t3 - t1
            tbar.set_description(f"Epoch: {e // self.batch_count}")

            if (e + 1) % self.params["val_check_interval"] == 0:
                t = time.time()
                self.model.eval()
                with torch.inference_mode():
                    val_acc, val_loss = self.evaluate(self.val_dataloader, "val")
                t_ = time.time()

                total_time += t_ - t

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

        print('Training has finished')
        print("Best val accuracy is", best_val_acc)
        wandb.run.summary["best_val_acc"] = best_val_acc

        t = time.time()
        self.model.eval()
        with torch.inference_mode():
            test_acc, test_loss = self.evaluate(self.test_dataloader, "test")
        t_ = time.time()

        total_time += t_ - t

        print("Total time taken for training and evaluation is", convert_time(total_time)) # format time in hh:mm:ss
        print("Final Test accuracy is", test_acc)
        print("--------Finish------------")

        wandb.run.summary["final_test_acc"] = test_acc
        wandb.run.summary["final_test_loss"] = test_loss

        wandb.run.summary["total_time"] = convert_time(total_time)
        wandb.finish()
        return test_acc, test_loss

    def evaluate(self, dataloader, mode="test"):
        loss = 0.0
        correct_all = 0
        total_all = 0
        for batch in tqdm(dataloader, desc=f"Evaluation Mode-{mode}"):
            for key in batch:
                batch[key] = batch[key].to(device=self.device)

            out = self.model(**batch)

            loss += self.loss_fn(*out)
            correct, total = self.metric_fn(*out)

            correct_all += correct
            total_all += total

        # mask = self.data.val_mask if mode == "val" else self.data.test_mask
        # mask = torch.nonzero(mask).squeeze() if mask.dtype == torch.bool else mask
        return (correct_all / total_all).item(), loss.item()