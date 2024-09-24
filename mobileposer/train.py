import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List
import lightning as L
from tqdm import tqdm 


class PoseDataset(Dataset):
    def __init__(self):
        pass