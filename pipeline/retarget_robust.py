"""
Mathematically Sound Delta-Rotation Retargeter for Blender 5.2.
Maps BVH motion directly onto character FBX bones via per-bone rest-pose transformation matrices:
R_target_local = (Target_Rest_World)^-1 * BVH_Pose_World * (BVH_Rest_World)^-1 * Target_Rest_World
Downscales textures to 1024px for lightweight 4MB GLBs.
"""

import bpy
import sys
import argparse
import os
import math
from mathutils import Matrix, Quaternion

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Delta Retarget BVH to Character FBX.")
    parser.add_argument("--character", required=True, help="Path to character FBX")
    parser.add_argument("--bvh", required=True, help="Path to BVH animation")
    parser.add_argument("--output", required=True, help="Path to output GLB")
    parser.add_argument("--anim_name", default="sign_animation", help="Animation clip name")
    return parser.parse_args(argv)

def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def downscale_textures():
    for img in bpy.data.images:
        if img.size[0] > 1024 or img.size[1] > 1024:
            new_w = min(1024, img.size[0])
            new_h = min(1024, img.size[1])
            img.scale(new_w, new_h)

def main():
    args = parse_args()
    clean_scene()
    
    # 1. Import Character FBX
    print(f"[1/5] Importing Character: {args.character}")
    bpy.ops.import_scene.fbx(filepath=args.character)
    
    char_arm = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            char_arm = obj
            break
    if not char_arm:
        raise RuntimeError("No armature found in character FBX!")
    char_arm.name = "Character_Armature"
    downscale_textures()
    
    # 2. Import BVH
    print(f"[2/5] Importing BVH: {args.bvh}")
    bpy.ops.import_anim.bvh(
        filepath=args.bvh,
        target='ARMATURE',
        global_scale=1.0,
        update_scene_fps=True,
        update_scene_duration=True
    )
    
    bvh_arm = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj != char_arm:
            bvh_arm = obj
            break
    if not bvh_arm:
        raise RuntimeError("No armature found in BVH import!")
        
    bvh_action = bvh_arm.animation_data.action if bvh_arm.animation_data else None
    frame_start = int(bvh_action.frame_range[0]) if bvh_action else 1
    frame_end = int(bvh_action.frame_range[1]) if bvh_action else 60
    
    # Build bone mapping dictionary
    char_bones = {}
    for pb in char_arm.pose.bones:
        clean = pb.name
        for prefix in ["mixamorig2:", "mixamorig:", "mixamorig_"]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        char_bones[clean.lower()] = pb

    # 3. Create fresh Animation Action on Character Armature
    if not char_arm.animation_data:
        char_arm.animation_data_create()
        
    target_action = bpy.data.actions.new(name=args.anim_name)
    char_arm.animation_data.action = target_action
    
    # Map of pairs: (char_pose_bone, bvh_pose_bone)
    bone_pairs = []
    for bvh_pb in bvh_arm.pose.bones:
        clean_bvh = bvh_pb.name.lower()
        if clean_bvh in char_bones:
            char_pb = char_bones[clean_bvh]
            bone_pairs.append((char_pb, bvh_pb))
            
    print(f"[3/5] Transferring Delta Rotations across {len(bone_pairs)} bones for {frame_end - frame_start + 1} frames...")
    
    # Pre-calculate Rest-Pose World Transformation Matrices
    # For every bone: Target_Rest_World & BVH_Rest_World
    rest_matrices = {}
    for char_pb, bvh_pb in bone_pairs:
        # Bone rest matrix in world space:
        m_rest_world = char_arm.matrix_world @ char_pb.bone.matrix_local
        b_rest_world = bvh_arm.matrix_world @ bvh_pb.bone.matrix_local
        
        # Delta matrix between BVH rest and Target rest:
        # D = (b_rest_world)^-1 @ m_rest_world
        delta_rest = b_rest_world.inverted() @ m_rest_world
        rest_matrices[char_pb.name] = (m_rest_world, delta_rest)

    # 4. Keyframe each frame
    scene = bpy.context.scene
    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        
        for char_pb, bvh_pb in bone_pairs:
            # Current animated world matrix of BVH bone:
            b_pose_world = bvh_arm.matrix_world @ bvh_pb.matrix
            
            m_rest_world, delta_rest = rest_matrices[char_pb.name]
            
            # Reconstruct true target world matrix without roll distortions:
            m_target_world = b_pose_world @ delta_rest
            
            # Convert target world matrix to pose bone local matrix:
            if char_pb.parent:
                m_parent_world = char_arm.matrix_world @ char_pb.parent.matrix
                m_local = m_parent_world.inverted() @ m_target_world
            else:
                m_local = char_arm.matrix_world.inverted() @ m_target_world
                
            # Extract local rotation quaternion:
            char_pb.rotation_mode = 'QUATERNION'
            char_pb.rotation_quaternion = m_local.to_quaternion()
            char_pb.keyframe_insert(data_path="rotation_quaternion", frame=f)
            
            if not char_pb.parent:
                # Set Hips position
                char_pb.location = m_local.to_translation()
                char_pb.keyframe_insert(data_path="location", frame=f)

    # 5. Clean up & Export
    bpy.data.objects.remove(bvh_arm, do_unlink=True)
    if bvh_action:
        bpy.data.actions.remove(bvh_action)
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format='GLB',
        export_animations=True,
        export_current_frame=False,
        export_skins=True,
        export_morph=False,
        export_lights=False,
        export_apply=False,
        export_image_format='JPEG',
        export_jpeg_quality=85
    )
    
    print(f"SUCCESS: Exported True-Delta GLB to: {args.output}")

if __name__ == "__main__":
    main()
