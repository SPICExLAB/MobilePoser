import torch
import os
import pickle
import numpy as np
import datetime
from typing import List, Tuple, Optional

from config import paths
from mobileposer.viewer.joint_viewer import JointViewer
from mobileposer.viewer.smpl_viewer import SMPLViewer
import mobileposer.articulate as art
from mobileposer.models.poser import Poser
from mobileposer.transpose.net import TransPoseNet

from external.config import *
from external.IMUPoser import IMUPoserModel

# Constants
NAME = "Vasco"
BASE_PATH = f"{NAME}_data"
JOINTS_IGNORED = [0, 7, 8, 10, 11, 20, 21, 22, 23]
JOINTS_REDUCED = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SMPL_QUEST_INDICES = torch.tensor([1, 3, 4, 5, 6, 9, 14, 7, 8, 13, 11, 16, 19, 45, 18, 44])
SMPL_UPPER = torch.tensor([0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
SMPL_LOWER = torch.tensor([1, 2, 4, 5, 8, 7, 10, 11])

def calibrate_imu_to_smpl(calib_acc: torch.Tensor, calib_ori: torch.Tensor, calib_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform calibration."""
    global_orient = calib_pose[0]  # Global orientation is the first joint
    R_imu_to_smpl = global_orient @ calib_ori.t()
    gravity_magnitude = 9.81  # m/s^2
    gravity_smpl = torch.tensor([0, -gravity_magnitude, 0], device=calib_acc.device)
    acc_offset = gravity_smpl - R_imu_to_smpl @ calib_acc
    return R_imu_to_smpl, acc_offset

def match_imu_data(xr_ts: torch.Tensor, imu_ts: torch.Tensor, calib_point: int):
    """Find closest IMU value to current Quest data."""
    xr_timestamp = xr_ts[calib_point]
    time_diffs = torch.abs(imu_ts - xr_timestamp)
    closest_imu_index = torch.argmin(time_diffs).item()
    closest_imu_timestamp = imu_ts[closest_imu_index]
    return closest_imu_index, closest_imu_timestamp, xr_timestamp

def unix_to_readable(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S.%f')

def transform_to_relative(quest_joints: torch.Tensor) -> torch.Tensor:
    """Transform joints to be relative to the root joint."""
    root_index = 1
    root_joint_coords = quest_joints[:, root_index, :]
    return quest_joints - root_joint_coords.unsqueeze(1)

def reduced_pose_to_full(reduced_pose: torch.Tensor) -> torch.Tensor:
    """Transform reduced pose to full pose."""
    B, S = reduced_pose.shape[:2]
    reduced_pose = reduced_pose.view(B, S, 16, 3, 3)
    full_pose = torch.eye(3, device=reduced_pose.device).repeat(B, S, 24, 1, 1)
    full_pose[:, :, JOINTS_REDUCED] = reduced_pose
    return full_pose.view(B, S, -1)

def reduced_global_to_full(reduced_pose: torch.Tensor, bodymodel: art.model.ParametricModel) -> torch.Tensor:
    """Convert reduced global pose to full pose."""
    pose = art.math.angular.r6d_to_rotation_matrix(reduced_pose).view(-1, 16, 3, 3)
    pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
    pred_pose = bodymodel.inverse_kinematics_R(pose)
    for ignore in JOINTS_IGNORED:
        pred_pose[:, ignore] = torch.eye(3, device=DEVICE)
    pred_pose[:, 0] = pose[:, 0]
    return pred_pose

class DataReader:
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def read_data(self, number: int, data_type: str) -> Optional[object]:
        number_str = f"{number:02d}"
        file_pattern = f"{NAME}_{number_str}_{data_type}"
        for filename in os.listdir(self.base_path):
            if filename.startswith(file_pattern) and filename.endswith(".pkl"):
                file_path = os.path.join(self.base_path, filename)
                with open(file_path, 'rb') as file:
                    return pickle.load(file)
        return None

def map_and_transform_to_smpl(quest_joints: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
    """Map and transform Quest joints to SMPL format."""
    quest_joints_relative = transform_to_relative(quest_joints)
    smpl_joints_relative = torch.zeros((quest_joints.size(0), 24, 3), device=quest_joints.device, dtype=quest_joints.dtype)
    zj_repeated = zj.repeat(quest_joints.size(0), 1, 1).double()
    smpl_joints_relative[:, SMPL_UPPER] = quest_joints_relative[:, SMPL_QUEST_INDICES]
    smpl_joints_relative[:, SMPL_LOWER] = zj_repeated[:, SMPL_LOWER]
    return smpl_joints_relative

def safe_convert(x: np.ndarray) -> np.ndarray:
    """Safely convert numpy arrays to float64."""
    if isinstance(x, np.ndarray) and x.size == 1:
        return np.float64(x)
    elif isinstance(x, (list, np.ndarray)):
        return np.array([safe_convert(i) for i in x], dtype=np.float64)
    else:
        try:
            return np.float64(x)
        except (ValueError, TypeError):
            return np.nan

def main():
    print("Starting main...")

    poser = Poser.from_pretrained('poser.ckpt')
    bodymodel = art.model.ParametricModel(paths.smpl_file, device=DEVICE)
    zj, _ = bodymodel.get_zero_pose_joint_and_vertex()
    zj = zj.unsqueeze(0)

    # read data file
    number = 55
    reader = DataReader(base_path=BASE_PATH)
    xr_data = reader.read_data(number, "xrhands")
    imu_data = reader.read_data(number, "imu")

    print("Quest Rotation Data")
    print(xr_data.rotation_data[0].shape)

    # read imu data
    imu_ts = torch.from_numpy(imu_data.timestamp.to_numpy())
    imu_data = imu_data.data.to_numpy().flatten()
    imu_data = safe_convert(imu_data)
    imu_data = torch.from_numpy(imu_data)

    # read acceleration and orientation data
    acc_raw = imu_data[:, 0:3]
    ori_raw = art.math.quaternion_to_rotation_matrix(imu_data[:, 9:13]).view(-1, 3, 3)
    print(f"Acc. Shape: {acc_raw.shape}")
    print(f"Ori. Shape: {ori_raw.shape}")
    
    # Quest joint position data
    xr_ts = torch.from_numpy(xr_data.timestamp.to_numpy()) 
    points = xr_data.position_data.to_numpy().flatten()
    points = safe_convert(points)
    quest_joints = torch.from_numpy(points)
    print(quest_joints[:, 0])
    print(f"Quest Joints: {quest_joints.shape}")

    # Find the corresponding IMU data for the CALIB_POINT
    CALIB_POINT = 450
    imu_idx, imu_timestamp, xr_timestamp = match_imu_data(xr_ts, imu_ts, CALIB_POINT)
    xr_time_human = unix_to_readable(xr_timestamp.item())
    imu_time_human = unix_to_readable(imu_timestamp.item())
    time_difference = (imu_timestamp - xr_timestamp).item() * 1000  # convert to milliseconds
    print(f"XR Timestamp at CALIB_POINT: {xr_time_human}")
    print(f"Corresponding IMU Timestamp: {imu_time_human}")
    print(f"Time Difference: {time_difference:.3f} ms")

    quest_joints = quest_joints[500:] # use a subset for testing
    quest_joints[:, :, 0] = -quest_joints[:, :, 0] # flip the x-coordinates

    joints = map_and_transform_to_smpl(quest_joints, zj)

    joints = joints[:500].reshape(1, -1, 72)
    pose = poser(joints.float(), [500])
    pose = reduced_global_to_full(pose, bodymodel)

    # grab IMU data
    calib_acc = acc_raw[imu_idx]
    calib_ori = ori_raw[imu_idx]
    calib_pose = pose[CALIB_POINT].double()
    calib_pose = calib_pose[19] # right elbow

    # manual calibration
    smpl2imu = torch.eye(3).t().double() # ORIENTATION OF THE PELVIS
    device2bone = smpl2imu.matmul(calib_ori).transpose(0, 1).matmul(calib_pose)
    acc_offsets = smpl2imu.matmul(calib_acc.unsqueeze(-1))  

    glb_acc = (smpl2imu.matmul(acc_raw.view(-1, 3, 1)) - acc_offsets).view(-1, 3)
    glb_ori = smpl2imu.matmul(ori_raw).matmul(device2bone)    
    print(f"Glb acc. : {glb_acc.shape}")
    print(f"Glb ori. : {glb_ori.shape}")

    # run through IMUPoser
    combo2imuidx = {
        "lw": 0,
        "rw": 1,
        "lp": 2,
        "rp": 3,
        "h": 4,
    }
    combo_id = "rw"
    imu_ids = [combo2imuidx[x] for x in combo_id.split("_")] 

    """
    # model = IMUPoserModel(config=pretrained_config)
    # checkpoint = torch.load("./external/checkpoints/epoch=epoch=81-val_loss=validation_step_loss=0.00973.ckpt", map_location=torch.device('cpu'))['state_dict']
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=" * 10)
    pretrained_config = Config(model="GlobalModelIMUPoser", project_root_dir="../IMUPoser", joints_set=amass_combos[combo_id], normalize="no_translation", r6d=True, use_acc_recon_loss=False, use_joint_loss=True, use_vposer_loss=False, use_vel_loss=False)
    pretrained_model = IMUPoserModel.load_from_checkpoint("./external/checkpoints/epoch=epoch=81-val_loss=validation_step_loss=0.00973.ckpt", config=pretrained_config, strict=False)
    seq_length = glb_acc.shape[0]
    _combo_acc = torch.zeros((3,)).repeat(seq_length, 5, 1)
    _combo_ori = torch.zeros((3, 3)).repeat(seq_length, 5, 1, 1)

    _combo_acc[:, 1, :] = glb_acc
    _combo_ori[:, 1, :] = glb_ori
    imu_input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1)

    pretrained_model.eval()
    with torch.no_grad():
        pred_pose = pretrained_model(imu_input.unsqueeze(0), [imu_input.shape[0]]).squeeze(0)    
    
    print(pred_pose.shape)
    """

    viewer = JointViewer(quest_joints)
    viewer.view()

    # viewer = SMPLViewer()
    # viewer.view(pose.detach())

if __name__ == "__main__":
    main()