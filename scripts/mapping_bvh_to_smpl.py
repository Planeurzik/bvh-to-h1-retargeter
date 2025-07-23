import numpy as np
from pathlib import Path

from scipy.spatial.transform import Rotation as R

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

BVH_JOINT_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase"
]

BVH_TO_SMPL = {
    "Hips": "pelvis",
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",
    "Spine": "spine_1",
    "Spine1": "spine_2",
    "Neck": "neck",
    "Head": "head",
    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",
    "LeftFoot": "left_ankle",
    "RightFoot": "right_ankle",
    "LeftToeBase": "left_big_toe",
    "RightToeBase": "right_big_toe",
    "LeftShoulder": "left_collar",
    "RightShoulder": "right_collar",
    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",
    "LeftHand": "left_wrist",
    "RightHand": "right_wrist",
    "LeftHandThumb1": "left_thumb",
    "LeftHandIndex1": "left_index",
    "LeftHandMiddle1": "left_middle",
    "LeftHandRing1": "left_ring",
    "LeftHandPinky1": "left_pinky",
    "RightHandThumb1": "right_thumb",
    "RightHandIndex1": "right_index",
    "RightHandMiddle1": "right_middle",
    "RightHandRing1": "right_ring",
    "RightHandPinky1": "right_pinky",
}

def extract_smpl_keypoints(world_data, bvh_joint_names, smpl_joint_names, bvh_to_smpl):
    """
    world_data: (T, 51, 7) with (x, y, z, qw, qx, qy, qz)
    bvh_joint_names: list of 51 joint names
    smpl_joint_names: list of 45 SMPL joint names
    bvh_to_smpl: dict mapping BVH names to SMPL names
    """
    T = world_data.shape[0]
    smpl_keypoints = np.zeros((T, len(smpl_joint_names), 3))

    smpl_to_bvh = {smpl: bvh for bvh, smpl in bvh_to_smpl.items()}

    for i, smpl_name in enumerate(smpl_joint_names):
        if smpl_name in smpl_to_bvh:
            bvh_name = smpl_to_bvh[smpl_name]
            if bvh_name in bvh_joint_names:
                bvh_idx = bvh_joint_names.index(bvh_name)
                smpl_keypoints[:, i, :] = world_data[:, bvh_idx, :3] 
            else:
                smpl_keypoints[:, i, :] = np.nan
        else:
            smpl_keypoints[:, i, :] = np.nan

    return smpl_keypoints

def widen_feet_laterally_orientation_aware(
    smpl_kps: np.ndarray,
    pelvis_rot_quat: np.ndarray,
    delta: float = 0.05,
) -> np.ndarray:
    """
    Push left and right foot/leg keypoints apart laterally in pelvis-local frame.

    Args:
        smpl_kps: (T, 45, 3) SMPL keypoints in world space.
        pelvis_rot_quat: (T, 4) quaternion [w, x, y, z] for pelvis orientation.
        delta: lateral offset per side (meters).

    Returns:
        smpl_kps_modified: (T, 45, 3) modified keypoints.
    """
    T = smpl_kps.shape[0]
    smpl_kps_modified = smpl_kps.copy()

    LEFT_INDICES = [
        SMPL_JOINT_NAMES.index(name) for name in [
            "left_ankle", "left_foot", "left_big_toe", "left_small_toe", "left_heel", "left_knee"
        ]
    ]
    RIGHT_INDICES = [
        SMPL_JOINT_NAMES.index(name) for name in [
            "right_ankle", "right_foot", "right_big_toe", "right_small_toe", "right_heel", "right_knee"
        ]
    ]

    pelvis_idx = SMPL_JOINT_NAMES.index("pelvis")
    pelvis_pos = smpl_kps[:, pelvis_idx, :]

    for t in range(T):
        quat_wxyz = pelvis_rot_quat[t]
        quat_xyzw = np.roll(quat_wxyz, -1)  # convert wxyz -> xyzw
        R_world_pelvis = R.from_quat(quat_xyzw).as_matrix()
        right_vector = R_world_pelvis[:, 1]  # local Y-axis of the pelvis in world frame

        # Apply lateral shift
        smpl_kps_modified[t, LEFT_INDICES] += delta * right_vector
        smpl_kps_modified[t, RIGHT_INDICES] -= delta * right_vector

    return smpl_kps_modified


asset_dir = Path(__file__).parent / ".." / "data_folder"

world_data = np.load(asset_dir / "joint_positions.npy")

world_data = world_data[::5]
#world_data = world_data[:100]

smpl_kps = extract_smpl_keypoints(world_data, BVH_JOINT_NAMES, SMPL_JOINT_NAMES, BVH_TO_SMPL)

pelvis_index = 0

pelvis = smpl_kps[:, pelvis_index:pelvis_index+1, :]

pelvis_quats = world_data[:, pelvis_index, 3:7]

scale_factors = np.array([1.0, 1.0, 1.1])
smpl_scaled = (smpl_kps - pelvis) * scale_factors + pelvis

smpl_scaled = widen_feet_laterally_orientation_aware(smpl_scaled, pelvis_rot_quat = pelvis_quats, delta = 0.1)

print("Scales:", scale_factors)

np.save(asset_dir / "smpl_keypoints.npy", smpl_scaled)
