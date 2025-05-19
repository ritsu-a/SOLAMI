import os
import numpy as np 
import sys
sys.path.append('SOLAMI_data/HumanTOMATO/src/tomato_represenation')
sys.path.append('tools/smplx')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/motiongpt/mGPT/data/humanml')
import smplx
import roma
import argparse
import copy
from os.path import join as pjoin

from common.skeleton import Skeleton
from common.quaternion import *
from utils.paramUtil import *
import torch
import torch.nn.functional as F

male_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_MALE.npz'
female_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_FEMALE.npz'
neutral_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_NEUTRAL.npz'

model_paths = {
    'male': male_model_path,
    'female': female_model_path,
    'neutral': neutral_model_path,
}

betas = np.array([-0.06134899, -0.4861751 ,  0.8630473 , -3.07320443,  1.10772016,
       -1.44656493,  2.97690664, -1.12731489,  1.24817344, -1.4111463 ,
       -0.04035034, -0.29547926,  0.38509519,  0.13750311,  0.94445029,
       -0.47172116])

parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
        53]


gender = 'male'


def cont6d_to_nearest_rotmat(cont6d):
    v1 = cont6d[..., :3]  # B * L * 3
    v2 = cont6d[..., 3:]  # B * L * 3

    u1 = F.normalize(v1, dim=-1)  # B * L * 3

    # norm_v1 = torch.norm(v1, p=2, dim=-1, keepdim=True)

    # u1 = v1 / norm_v1  # B * L * 3
    dot_product = torch.sum(u1 * v2, dim=-1, keepdim=True)  # B * L * 1
    proj_v2_on_u1 = dot_product * u1  # B * L * 3
    u2 = F.normalize(v2 - proj_v2_on_u1, dim=-1)  # B * L * 3
    # norm_v2 = torch.norm(v2 - proj_v2_on_u1, p=2, dim=-1, keepdim=True)
    # u2 = (v2 - proj_v2_on_u1) / norm_v2
    u3 = torch.cross(u1, u2, dim=-1)  # B * L * 3
    rotation_matrix = torch.stack([u1, u2, u3], dim=-1)  # B * L * 3 * 3
    
    return rotation_matrix


def cont6d_to_nearest_rotvec(cont6d):
    rotation_matrix = cont6d_to_nearest_rotmat(cont6d)
    rotation_vector = roma.rotmat_to_rotvec(rotation_matrix)
    return rotation_vector, rotation_matrix



def process_file(data, feet_thre):
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # body,hand joint idx
    # 2*3*5=30, left first, then right
    hand_joints_id = [i for i in range(25, 55)]
    body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 22 joints
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 52
    
    # Uniformly scale the skeleton to match a target offset
    positions = data.copy()
    if positions.shape[-2] == 55:
        positions = positions[:, body_joints_id+hand_joints_id, :]
    origin_quat, origin_root = get_all_original_root_rot(positions)
    positions, scale = uniform_skeleton(positions, tgt_offsets)
    
    # Put the skeleton on the floor by subtracting the minimum height
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # Ensure the initial facing direction is along Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    # Rotate the motion capture data using the calculated quaternion
    # positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)
    
    # Store the global positions for further analysis
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array(
            [thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z)
                  < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z)
                  < velfactor)).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        """
        Adjusts the motion capture data to a local pose representation and ensures
        that all poses face in the Z+ direction.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            numpy.ndarray: Adjusted motion capture data in a local pose representation.
        """
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        """
        Computes quaternion parameters, root linear velocity, and root angular velocity
        based on the input motion capture data.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            tuple: A tuple containing quaternion parameters, root angular velocity, root linear velocity, and root rotation.
        """
        # Initialize a skeleton object with a specified kinematic chain
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        """
        Computes continuous 6D parameters, root linear velocity, and root angular velocity
        based on the input motion capture data.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            tuple: A tuple containing continuous 6D parameters, root angular velocity, root linear velocity, and root rotation.
        """
        # Initialize a skeleton object with a specified kinematic chain
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot
    
    # Extract additional features including root height and root data
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Get Joint Rotation Representation
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Get Joint Rotation Invariant Position Represention
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Get Joint Velocity Representation
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    # Concatenate all features into a single array
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity, (origin_quat, origin_root, scale, root_quat_init)



def recover_root_rot_pos(data):
    """
    Recover root rotation and position from the given motion capture data.

    Args:
        data (torch.Tensor): Input motion capture data with shape (..., features).

    Returns:
        tuple: A tuple containing the recovered root rotation quaternion and root position.
    """
    # Extract root rotation velocity from the input data
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=52):
    # Recover root rotation quaternion and position from the input data
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(
        positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def recover_from_smplx_feature(data_, rot_type='local'):
    # motion_len = len(data)
    # root_rotmat = cont6d_to_matrix(torch.Tensor(smplx_features['global_root_cont6d']))
    # root_rotvec = roma.rotmat_to_rotvec(root_rotmat)
    if type(data_) == np.ndarray:
        data = torch.tensor(data_, dtype=torch.float32)
    else:
        data = data_
    root_rotvec, root_rotmat = cont6d_to_nearest_rotvec(data[..., 3:9])
    root_rotmat_y = roma.rotmat_to_euler('yzx', root_rotmat)
    root_rotmat_y[..., 1:] *= 0
    root_rotmat_y = roma.euler_to_rotmat('yzx', root_rotmat_y)
    
    root_vel_xz = torch.zeros_like(data[..., 0:3]).to(data.device)
    root_vel_xz[..., [0, 2]] += data[..., 0:2]
    root_xz_vel = torch.matmul(root_rotmat_y, root_vel_xz.unsqueeze(-1)).squeeze(-1)
    
    root_pos_xz = torch.zeros_like(data[..., 0:3]).to(data.device)
    root_pos_xz[..., 1:, [0, 2]] += root_xz_vel[..., :-1, [0, 2]]
    root_pos_xz = torch.cumsum(root_pos_xz, dim=-2)
    root_pos_xz[..., 1:2] += data[..., 2:3]
    
    rot_vec_cont6d = data[..., 9:]
    rot_vec_cont6d = rot_vec_cont6d.view(rot_vec_cont6d.shape[:-1] + (-1, 6))
    rot_vec, rot_mat = cont6d_to_nearest_rotvec(rot_vec_cont6d)
    
    if rot_type == 'global':
        
        parents_len = int(data[..., 9:].shape[-1] / 6) + 1
        parents_ = parents[:parents_len]
        
        init_pose = torch.eye(3).repeat((rot_mat.shape[:-3]) + (1, 1, 1)).to(data.device)
        rotmat_global = torch.cat([init_pose, rot_mat], dim=-3)
        rotmat_local = []
        for i in range(1, len(parents_)):
            current_res = torch.matmul(roma.rotmat_inverse(rotmat_global[..., parents_[i], :, :]), rotmat_global[..., i, :, :])
            rotmat_local.append(current_res)
        rotmat_local = torch.stack(rotmat_local, dim=-3)
        rot_vec_final = roma.rotmat_to_rotvec(rotmat_local)
        rot_vec_final = rot_vec_final.view(rot_vec_final.shape[:-2] + (-1,))
        data_recover = torch.cat([root_pos_xz, root_rotvec, rot_vec_final], dim=-1)
    else:
        rot_vec_final = rot_vec.view(rot_vec.shape[:-2] + (-1,))
        data_recover = torch.cat([root_pos_xz, root_rotvec, rot_vec_final], dim=-1)
    return data_recover


def process_smplx_feature(data, rot_type='local'):
    # lose 1 timestamp
    root_velocity = (data[..., 1:, :3] - data[..., :-1, :3])
    root_rotation_euler = roma.rotvec_to_euler('yzx', data[..., 3:6].clone())
    root_rotation_euler[..., 1:] *= 0
    root_rotmat_y = roma.euler_to_rotmat('yzx', root_rotation_euler)
    root_rotmat_inv = roma.rotmat_inverse(root_rotmat_y)
    root_velocity = root_rotmat_inv[..., :-1, :, :] @ root_velocity.unsqueeze(-1)
    root_velocity_xz = root_velocity[..., [0, 2], 0]
    
    ### root height y
    root_height = data[... , 1:2]
    global_orient = roma.rotvec_to_rotmat(data[..., 3:6])
    global_root_cont6d = torch.cat([global_orient[..., 0], global_orient[..., 1]], dim=-1)
    
    rot_rotmat = roma.rotvec_to_rotmat(data[..., 6:].view(data.shape[:-1] + (-1, 3)))
    if rot_type == 'local':
        rot_cont6d = torch.cat([rot_rotmat[..., 0], rot_rotmat[..., 1]], dim=-1)
    else:
        global_rotmat = []
        
        parents_len = int(data[..., 6:] / 3) + 1
        parents_ = parents[:parents_len]
        
        global_rotmat.append(torch.eye(3).repeat(rot_rotmat.shape[:-3], 1, 1).to(data.device))
        for i in range(1, len(parents)):
            current_res = torch.matmul(global_rotmat[parents[i]], rot_rotmat[..., i, :, :])
            global_rotmat.append(current_res)
        global_rotmat = torch.stack(global_rotmat, dim=-3)
        rot_cont6d = torch.cat([global_rotmat[..., 0], global_rotmat[..., 1]], dim=-1)
        rot_cont6d = rot_cont6d[..., 1:, :, :]
    
    rot_cont6d = rot_cont6d.view(rot_cont6d.shape[:-2] + (-1,))
    
    data_recover = torch.cat([root_velocity_xz, root_height[..., :-1, :], 
                              global_root_cont6d[..., :-1, :], rot_cont6d[..., :-1, :]], dim=-1)
    return data_recover


def preprocess_smplx(smplx_joints, root_rotvec, root_transl, t_root_J):
    # Uniformly scale the skeleton to match a target offset
    device = smplx_joints.device
    batch_size, seq_len, joints_num, _ = smplx_joints.shape
    smplx_joints = smplx_joints.view(-1, joints_num, 3)
    
    hand_joints_id = [i for i in range(25, 55)]
    body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    
    positions = smplx_joints[..., body_joints_id+hand_joints_id, :].cpu().numpy()
    
    origin_root = positions[:, 0]
    
    # Put the skeleton on the floor by subtracting the minimum height
    floor_height = positions.min(axis=0).min(axis=0)[1]
    # positions[:, :, 1] -= floor_height

    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    root_pose_init_xz[1] = floor_height
    # positions = positions - root_pose_init_xz
    
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    r_hip, l_hip = 2, 1

    # Ensure the initial facing direction is along Z+
    root_pos_init = positions[0]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    
    origin_quat = root_quat_init.copy()
    
    root_quat_init = np.ones(positions.shape[:1] + (4,)) * root_quat_init

    # # Rotate the motion capture data using the calculated quaternion
    # positions = qrot_np(root_quat_init, positions)
    smplx_orient = root_rotvec.clone()
    smplx_orient_mat = roma.rotvec_to_rotmat(torch.Tensor(smplx_orient).to(device))
    
    # apply transform on smplx parameters
    euler_transfer = qeuler_np(root_quat_init, 'yzx') / 180 * np.pi
    # since forward init vector is set in xz plane, so only rotate y axis!
    transfer_mat = roma.euler_to_rotmat('yzx', torch.Tensor(euler_transfer).to(device))
    smplx_orient_modified_mat = torch.matmul(transfer_mat, smplx_orient_mat)
    smplx_orient_modified = roma.rotmat_to_rotvec(smplx_orient_modified_mat)

    
    # root_pose_init_xz_new = (transfer_mat[0] @ torch.Tensor(root_pose_init_xz)).numpy()
    transl = root_transl.clone()
    transl_new =  (transfer_mat[0] @ (transl - torch.Tensor(root_pose_init_xz).to(device) + t_root_J).unsqueeze(-1)).squeeze(-1) - t_root_J
    # test
    return smplx_orient_modified, transl_new,  (origin_quat, origin_root)



def get_relative_pose(ske_forward_1,  ske_forward_2, scale_1, scale_2, ske_root_1, ske_root_2):
            
    quat_P1_in_P2 = qmul_np(qinv_np(ske_forward_1), ske_forward_2)
    pos_P1_in_P2 = qrot_np(ske_forward_2[:1], (scale_1 + scale_2) / 2. * (ske_root_1[:1] - ske_root_2[:1]))
    
    cont6d_P1_in_P2_save = quaternion_to_cont6d_np(quat_P1_in_P2[0])
    pos_P1_in_P2_save = pos_P1_in_P2[0]
    
    cont6d_P2_in_P1_save = quaternion_to_cont6d_np(qinv_np(quat_P1_in_P2[0]))
    pos_P2_in_P1_save = -qrot_np(qinv_np(quat_P1_in_P2[0]), pos_P1_in_P2[0])
    
    return cont6d_P1_in_P2_save, pos_P1_in_P2_save, cont6d_P2_in_P1_save, pos_P2_in_P1_save