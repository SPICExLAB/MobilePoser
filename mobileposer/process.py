import os
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import glob

from mobileposer.articulate.model import ParametricModel
from mobileposer.articulate import math
from mobileposer.constants import DataFold
from mobileposer.config import paths, datasets


# left wrist, right wrist, left thigh, right thigh, head, pelvis
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
body_model = ParametricModel(paths.smpl_file)

FPS = 25
AMASS_FOLDER = {60: "AMASS_60", 25: "AMASS"}


def _syn_acc(v, smooth_n=4):
    """Synthesize accelerations from vertex positions."""
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def process_amass():
    def _foot_ground_probs(joint):
        """Compute foot-ground contact probabilities."""
        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = (dist_lfeet < 0.008).int()
        rfoot_contact = (dist_rfeet < 0.008).int()
        lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
        rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)

    try:
        processed = [fpath.name for fpath in (paths.processed_datasets / AMASS_FOLDER[FPS]).iterdir()]
    except FileNotFoundError:
        processed = []
    processed = []

    for ds_name in datasets.amass_datasets:
        if ds_name in processed:
            continue
        data_pose, data_trans, data_beta, length = [], [], [], []
        print('\rReading', ds_name)

        for npz_fname in tqdm(sorted(glob.glob(os.path.join(paths.raw_amass, ds_name, '*/*_poses.npz')))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]     # right hand 
        pose = pose[:, :24].clone()   # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print('Synthesizing IMU accelerations and orientations')
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

            out_pose.append(pose[b:b + l].clone())  # N, 24, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_contact.append(_foot_ground_probs(joint).clone()) # N, 2

            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            b += l

        print('Saving')
        amass_dir = paths.processed_datasets / AMASS_FOLDER[FPS]
        amass_dir.mkdir(exist_ok=True, parents=True)
        ds_dir = amass_dir / ds_name
        ds_dir.mkdir(exist_ok=True)
        
        torch.save(out_pose, ds_dir / 'pose.pt')
        torch.save(out_shape, ds_dir / 'shape.pt')
        torch.save(out_tran, ds_dir / 'tran.pt')
        torch.save(out_joint, ds_dir / 'joint.pt')
        torch.save(out_vrot, ds_dir / 'vrot.pt')
        torch.save(out_vacc, ds_dir / 'vacc.pt')
        torch.save(out_contact, ds_dir / 'contact.pt')
        print('Synthetic AMASS dataset is saved at', str(ds_dir))


def process_totalcapture():
    """Preprocess TotalCapture dataset for testing."""

    inches_to_meters = 0.0254
    pos_file = 'gt_skel_gbl_pos.txt'
    ori_file = 'gt_skel_gbl_ori.txt'

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5']

    # Load poses from processed AMASS dataset
    amass_tc = torch.load(os.path.join(paths.processed_datasets, "AMASS", "TotalCapture", "pose.pt"))
    tc_poses = {pose.shape[0]: pose for pose in amass_tc}

    processed, failed_to_process = [], []
    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.calibrated_totalcapture)):
        if not file.endswith(".pkl") or ('s5' in file and 'acting3' in file) or not any(file.startswith(s.lower()) for s in subjects):
            continue

        data = pickle.load(open(os.path.join(paths.calibrated_totalcapture, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()
        acc = torch.from_numpy(data['acc']).float()

        # Load pose data from AMASS
        try: 
            name_split = file.split("_")
            subject, activity = name_split[0], name_split[1].split(".")[0]
            pose_npz = np.load(os.path.join(paths.raw_amass, "TotalCapture", subject, f"{activity}_poses.npz"))
            pose = torch.from_numpy(pose_npz['poses']).float().view(-1, 52, 3)
        except:
            failed_to_process.append(f"{subject}_{activity}")
            print(f"Failed to Process: {file}")
            continue

        pose = tc_poses[pose.shape[0]]
    
        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        # convert axis-angle to rotation matrix
        pose = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3, 3

        processed.append(file)
    
    for subject_name in subjects:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official, subject_name))):
            if (subject_name == 'S5' and motion_name == 'acting3') or motion_name.startswith(".") or (f"{subject_name.lower()}_{motion_name}" in failed_to_process):
                continue   # no SMPL poses

            f = open(os.path.join(paths.raw_totalcapture_official, subject_name, motion_name, pos_file))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    # remove acceleration bias
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = pose.view(-1, 24, 3, 3)
        _, _, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    os.makedirs(paths.processed_totalcapture, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.processed_totalcapture, f'totalcapture.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.processed_totalcapture)
    print(f'Files processed: {len(processed)}:', processed)


def process_imuposer(split: str="train"):
    """Preprocess the IMUPoser dataset"""

    train_split = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    test_split = ['P9', 'P10']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans = [], [], [], []
    for pid_path in sorted(paths.raw_imuposer.iterdir()):
        if pid_path.name not in subjects:
            continue

        print(f"Processing: {pid_path.name}")
        for fpath in sorted(pid_path.iterdir()):
            with open(fpath, "rb") as f: 
                fdata = pickle.load(f)
                
                acc = fdata['imu'][:, :5*3].view(-1, 5, 3)
                ori = fdata['imu'][:, 5*3:].view(-1, 5, 3, 3)
                pose = math.axis_angle_to_rotation_matrix(fdata['pose']).view(-1, 24, 3, 3)
                tran = fdata['trans'].to(torch.float32)
                
                 # align IMUPoser global fame with DIP
                rot = torch.tensor([[[-1, 0, 0], [0, 0, 1], [0, 1, 0.]]])
                pose[:, 0] = rot.matmul(pose[:, 0])
                tran = tran.matmul(rot.squeeze())

                # ensure sizes are consistent
                assert tran.shape[0] == pose.shape[0]

                accs.append(acc)    # N, 5, 3
                oris.append(ori)    # N, 5, 3, 3
                poses.append(pose)  # N, 24, 3, 3
                trans.append(tran)  # N, 3

    print(f"# Data Processed: {len(accs)}")
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.processed_datasets25, f'imuposer_{split}.pt'))


if __name__ == "__main__":
    # process_totalcapture()
    process_imuposer(split="train")
    # process_amass()