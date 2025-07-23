import bpy
import numpy as np
import os
from math import radians
from mathutils import Quaternion, Vector, Matrix

output_path = "~/Documents/joint_positions.npy"

output_path = os.path.expanduser(output_path)

armature = bpy.data.objects['walk']

joint_names = [bone.name for bone in armature.pose.bones]
num_joints = len(joint_names)

frame_start = bpy.context.scene.frame_start
frame_end = bpy.context.scene.frame_end
num_frames = frame_end - frame_start + 1

joint_positions = np.zeros((num_frames, num_joints, 7), dtype=np.float32)

for i, frame in enumerate(range(frame_start, frame_end + 1)):
    bpy.context.scene.frame_set(frame)
    for j, bone_name in enumerate(joint_names):
        bone = armature.pose.bones[bone_name]
        local_pos = bone.matrix.to_translation()
        
        m = bone.matrix_channel.to_3x3()
        
        correction = Quaternion((0.0, 0.0, 1.0), radians(-90))
        
        q_rel = m.to_quaternion()
        
        quat = correction @ q_rel
        
        joint_positions[i, j, :3] = [local_pos.x, local_pos.y, local_pos.z]
        joint_positions[i, j, 3:] = [quat.w, quat.x, quat.y, quat.z]


np.save(output_path, joint_positions)
print(f"Saved joint positions to: {output_path}")
