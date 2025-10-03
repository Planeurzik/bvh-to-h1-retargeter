import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

lin_vel_world = []
ang_vel_world = []
joint_vel = []
lin_vel_local = []
ang_vel_local = []

def compute_linear_velocities(positions):
    """Compute base linear velocities from positions in world frame."""
    return np.gradient(positions, axis=0) / dt

def compute_angular_velocities(quaternions):
    """Compute base angular velocities from quat orientations in world frame."""
    quat_diff = np.gradient(quaternions, axis=0) / dt

    angular_velocities = np.zeros((quaternions.shape[0], 3))
    for i in range(quaternions.shape[0]):
        Rmat = R.from_quat(quaternions[i]).as_matrix()
        q_dot = quat_diff[i]
        angular_velocities[i] = 2 * Rmat @ q_dot[:3]

    return angular_velocities

def transform_to_local_frame(lin_vel, ang_vel, quat):
    """Tranform base velocities from world to local base frame."""
    local_lin_vel = np.zeros_like(lin_vel)
    local_ang_vel = np.zeros_like(ang_vel)
    for i in range(quat.shape[0]):
        Rmat = R.from_quat(quat[i]).as_matrix()
        local_lin_vel[i] = Rmat.T @ lin_vel[i]
        local_ang_vel[i] = Rmat.T @ ang_vel[i]
    return local_lin_vel, local_ang_vel

csv_files = "data_folder/retargeted_trajectory.csv"
data = np.genfromtxt(csv_files, delimiter=",")
dt = 1.0 / 30  # Sequences are recorded at 30 Hz

data[:, :2] -= data[0, :2]

# print("Before --- (Yaw, Pitch, Roll)")
# print(quaternion_to_euler(sec[:, 3:7]))

YPR = R.from_quat(data[:, 3:7]).as_euler("zyx")
neg_yaw = R.from_euler("z", -YPR[0, 0])
data[:, :3] = neg_yaw.apply(data[:, :3])

# Remove yaw from orientation
YPR[:, 0] -= YPR[0, 0]
data[:, 3:7] = R.from_euler("zyx", YPR).as_quat()

# --------------------------------------- #
mirrored_data = np.copy(data)
# Mirror position
mirrored_data[:, 1] *= -1.0  # Y axis pos

# Mirror orientation
YPR = R.from_quat(mirrored_data[:, 3:7]).as_euler("zyx")
YPR[:, 0] *= -1.0  # yaw orientation
YPR[:, 2] *= -1.0  # roll orientation 
mirrored_data[:, 3:7] = R.from_euler("zyx", YPR).as_quat()

# Mirror joints
mirrored_data[:, 7:] = mirrored_data[:, 7:][:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 15, 16, 17, 18, 11, 12, 13, 14]]
mirrored_data[:, np.array([0, 5, 1, 6, 10, 12, 16, 13, 17]) + 7] *= -1.0

def compute_velocities(data):
    pos = data[:, :3]
    quat = data[:, 3:7]
    joint_pos = data[:, 7:]

    lin_vel_world = compute_linear_velocities(pos)
    ang_vel_world = compute_angular_velocities(quat)
    joint_vel = compute_linear_velocities(joint_pos)

    # Transform base velocities from world to base frame
    lin_vel_local, ang_vel_local = transform_to_local_frame(
        lin_vel_world, ang_vel_world, quat
    )
    return np.concatenate([lin_vel_local, ang_vel_local, joint_pos, joint_vel], axis=1)

new_data = compute_velocities(data)
mirror = compute_velocities(mirrored_data)
#result = np.concatenate([new_data[95:270], new_data[2700:3000], mirror[95:270], mirror[2700:3000]], axis=0)
#result = np.concatenate([new_data[120:725], mirror[120:725]], axis=0)
result = new_data[120:725]
#result = np.concatenate([new_data[300:5450], mirror[300:5450]], axis=0)

np.save('recordings/door_through.npy', result)