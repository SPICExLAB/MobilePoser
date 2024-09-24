import os
import numpy as np
import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mobileposer.models import *
from mobileposer.config import *
from mobileposer.viewers import SMPLViewer
from mobileposer.utils import load_model
from mobileposer.loader import DataLoader
import mobileposer.articulate as art


class Viewer:
    def __init__(self, dataset: str='imuposer', seq_num: int=0, combo: str='lw_rp'):
        """Viewer class for visualizing pose."""
        # load models 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.model = load_model(paths.weights_file).eval()
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)

        # setup dataloader
        self.dataloader = DataLoader(dataset, combo=combo)
        self.data = self.dataloader.load_data(seq_num)
    
    def _evaluate_model(self):
        """Evaluate the model."""
        data = self.data['imu']
        if getenv('ONLINE'):
            # online model evaluation (slower)
            pose, joints, tran, contact = [torch.stack(_) for _ in zip(*[self.model.forward_online(f) for f in tqdm(data)])]
        else:
            # offline model evaluation  
            with torch.no_grad():
                pose, joints, tran, contact = self.model.forward_offline(data.unsqueeze(0), [data.shape[0]]) 
        return pose, tran, joints, contact

    def view(self):
        """View the pose."""
        pose_t, tran_t = self.data['pose'], self.data['tran']
        pose_p, tran_p, _, _ = self._evaluate_model()
        viewer = SMPLViewer()
        viewer.view(pose_p, tran_p, pose_t, tran_t)
