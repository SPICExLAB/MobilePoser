import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser

from config import *
from viewer import Viewer
from utils import load_model


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--model', type=str, default=paths.weights_file)
    args.add_argument('--dataset', type=str, default='dev')
    args.add_argument('--seq-num', type=int, default=1)
    args = args.parse_args()

    # view dataset
    v = Viewer(dataset=args.dataset, seq_num=args.seq_num)
    v.view()

