import torch

from mobileposer.helpers import *
from mobileposer.config import *
import mobileposer.articulate as art


class SMPLViewer:
    def __init__(self, fps: int=25):
        self.fps = fps
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)

    def view(self, pose_p, tran_p, pose_t, tran_t, view_tran: bool=False): 
        if not view_tran:
            tran_p = torch.zeros(pose_p.shape[0], 3) # set translation to origin
        pose_p, tran_p = pose_p.view(-1, 24, 3, 3), tran_p.view(-1, 3)
        pose_t, tran_t = pose_t.view(-1, 24, 3, 3), tran_t.view(-1, 3)
        poses, trans = [pose_p], [tran_p]
        if getenv("GT") == 1: 
            # visualize pred & ground truth
            poses.append(pose_t)
            trans.append(tran_t)
        elif getenv("GT") == 2:
            # visualize ground truth only
            poses = [pose_t]
            trans = [tran_t]
        self.bodymodel.view_motion(poses, trans, fps=self.fps, distance_between_subjects=0)
