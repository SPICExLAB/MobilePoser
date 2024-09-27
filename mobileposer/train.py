import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List
import lightning as L
from tqdm import tqdm 


def train_module(module: str):
    # setup Wandb logger
    wandb_logger = WandbLogger(
        project=module_name, 
        name=get_datestring(),
        save_dir=module_path
    ) 



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--init-from", nargs="?", default="scratch", type=str)
    args = parser.parse_args()

    # set seed for reproducible results
    seed_everything(42, workers=True)




    # make 
