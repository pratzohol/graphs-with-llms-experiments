import pandas as pd
import numpy as np
import torch
from models.gnn import GNN
from models.mlp import MLP
from utils.dataloader import GetDataloader
from utils.args import get_params
from trainer import Trainer

from tqdm import trange
from tqdm import tqdm
import torch.nn.functional as F
import os
import os.path as osp
import yaml
import random
from datetime import date

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Using config.yaml to get params
    with open("config.yaml", "r") as f:
        params = yaml.safe_load(f)

    params["device"] = 'cpu' if params["device"] == 123 else f"cuda:{params['device']}"
    params["exp_name"] = f"{date.today()}. Experiment_{params['dataset']}_{params['sentence_encoder']}_{params['exp_name']}"

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    trnr = Trainer(params)
    trnr.train()
