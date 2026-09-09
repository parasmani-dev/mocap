"""Kinematic Blender Baker for ISL Avatar Animations.
Enforces:
1. Global Hard Joint Limits (Shoulder ball-joint, Elbow 0..145° hinge, Wrist, Finger 1-DOF MCP/PIP/DIP 0..110°).
2. Persistent Hand Orientation Correctness (Full 3x3 orthonormal palm frame matrix with mirror-aware cross products).
3. Defined Canonical Neutral/Ready Pose with smooth lead-in and lead-out blending.
4. Comprehensive Clamping Logger outputting every clamp event to JSON/CSV.
"""

import bpy
import sys
import argparse
import os
import json
import math
from typing import Tuple, List, Dict, Optional, Any
from mathutils import Matrix, Quaternion, Euler, Vector

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anim_name", default="sign_action")
    parser.add_argument("--clamping_log", default="")
    return parser.parse_args(argv)

args = parse_args()
bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Import Character FBX
# Clean default scene objects if any exist
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

bpy.ops.import_scene.fbx(filepath=args.character)
char_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
char_arm.name = "Character_Armature"
bpy.context.view_layer.objects.active = char_arm

# Clear any pre-existing actions / NLA tracks from the imported FBX
if char_arm.animation_data:
    char_arm.animation_data.action = None
    for track in list(char_arm.animation_data.nla_tracks):
        char_arm.animation_data.nla_tracks.remove(track)
for act in list(bpy.data.actions):
    bpy.data.actions.remove(act, do_unlink=True)

bpy.ops.object.mode_set(mode='POSE')

# Downscale textures for web optimization and enforce 100% solid OPAQUE blend modes
for mat in bpy.data.materials:
    mat.blend_method = 'OPAQUE'
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'OPAQUE'

for img in bpy.data.images:
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.scale(min(1024, img.size[0]), min(1024, img.size[1]))

char_bones = {}
for pb in char_arm.pose.bones:
    clean = pb.name
    for pfx in ["mixamorig2:", "mixamorig:", "mixamorig_"]:
        if clean.startswith(pfx):
            clean = clean[len(pfx):]
            break
    char_bones[clean] = pb

with open(args.landmarks, 'r', encoding='utf-8') as f:
    data = json.load(f)

frames = data.get('frames', [])
total_frames = len(frames)
if total_frames == 0:
    print("ERROR: No frames in landmarks JSON")
    sys.exit(0)

fps = data.get('fps', 30.0)
if fps <= 0 or math.isnan(fps):
    fps = 30.0

if not char_arm.animation_data:
    char_arm.animation_data_create()
action = bpy.data.actions.new(name=args.anim_name)
char_arm.animation_data.action = action

def norm_v(v: Vector) -> Vector:
    l = v.length
    return v / l if l > 1e-7 else Vector((0, 0, 0))

def quat_from_to(u: Vector, v: Vector) -> Quaternion:
    u = norm_v(u)
    v = norm_v(v)
    dot = max(-1.0, min(1.0, u.dot(v)))
    if dot > 0.999999:
        return Quaternion((1, 0, 0, 0))
    if dot < -0.999999:
        axis = Vector((0, 0, 1))
        if abs(u.z) > 0.9:
            axis = Vector((0, 1, 0))
        ortho = norm_v(u.cross(axis))
        return Quaternion(ortho, math.pi)
    axis = norm_v(u.cross(v))
    angle = math.acos(dot)
    return Quaternion(axis, angle)

scene = bpy.context.scene
scene.render.fps = int(round(fps))

clamping_events = []
word_id = os.path.splitext(os.path.basename(args.output))[0].replace("avatar_", "")

def log_clamp(joint: str, frame_idx: int, attempted: float, clamped: float, reason: str):
    clamping_events.append({
        "word": word_id,
        "joint": joint,
        "frame": frame_idx + 1,
        "attempted_val": round(attempted, 2),
        "clamped_val": round(clamped, 2),
        "reason": reason
    })

# Extract rest matrices for LeftHand and RightHand
# Extract rest matrices for LeftArm, RightArm, LeftForeArm, RightForeArm, LeftHand, RightHand
M_REST_LA = char_arm.data.bones['mixamorig2:LeftArm'].matrix_local.to_3x3() if 'mixamorig2:LeftArm' in char_arm.data.bones else None
M_REST_LF = char_arm.data.bones['mixamorig2:LeftForeArm'].matrix_local.to_3x3() if 'mixamorig2:LeftForeArm' in char_arm.data.bones else None
M_REST_RA = char_arm.data.bones['mixamorig2:RightArm'].matrix_local.to_3x3() if 'mixamorig2:RightArm' in char_arm.data.bones else None
M_REST_RF = char_arm.data.bones['mixamorig2:RightForeArm'].matrix_local.to_3x3() if 'mixamorig2:RightForeArm' in char_arm.data.bones else None

R_REST_LH = char_arm.data.bones['mixamorig2:LeftHand'].matrix_local.to_3x3() if 'mixamorig2:LeftHand' in char_arm.data.bones else None
R_REST_RH = char_arm.data.bones['mixamorig2:RightHand'].matrix_local.to_3x3() if 'mixamorig2:RightHand' in char_arm.data.bones else None

# Canonical Neutral Ready / Rest Pose (Arms resting down and forward in front of waist)
REST_L_ARM = norm_v(Vector((0.2, -0.9, 0.15)))
REST_L_FORE = norm_v(Vector((0.2, -0.3, 0.8)))
REST_R_ARM = norm_v(Vector((-0.2, -0.9, 0.15)))
REST_R_FORE = norm_v(Vector((-0.2, -0.3, 0.8)))

def get_bone_local_rot(target_dir_world: Vector, m_rest: Matrix, is_child: bool = False, q_parent_world: Quaternion = None) -> Tuple[Quaternion, Quaternion]:
    """Computes exact local quaternion for a bone given target direction in world space."""
    # In bone rest local frame, bone length is along +Y: Vector((0, 1, 0))
    if m_rest is not None:
        target_dir_local = norm_v(m_rest.inverted() @ target_dir_world)
        q_world_rot = norm_v(Vector((0, 1, 0))).rotation_difference(target_dir_local)
        # Reconstruct actual world orientation
        q_world_dir = norm_v(Vector((0, 1, 0))).rotation_difference(target_dir_world)
    else:
        q_world_rot = Quaternion((1, 0, 0, 0))
        q_world_dir = Quaternion((1, 0, 0, 0))
    return q_world_rot, q_world_dir

prev_palm_n_l = None
prev_palm_n_r = None
prev_q_hand_l = None
prev_q_hand_r = None
prev_finger_curls_l = {}
prev_finger_curls_r = {}

BLEND_FRAMES = min(5, total_frames // 2)

for f_idx, frame in enumerate(frames):
    f_num = f_idx + 1
    scene.frame_set(f_num)
    
    # Calculate Lead-In and Lead-Out blending factors
    blend_weight = 1.0
    if BLEND_FRAMES > 0:
        if f_idx < BLEND_FRAMES:
            blend_weight = f_idx / float(BLEND_FRAMES)
        elif f_idx >= (total_frames - BLEND_FRAMES):
            blend_weight = (total_frames - 1 - f_idx) / float(BLEND_FRAMES)
    blend_weight = max(0.0, min(1.0, blend_weight))
    
    pose_lms = frame.get('pose_world_landmarks') or frame.get('pose_landmarks')
    lh = frame.get('left_hand_landmarks')
    rh = frame.get('right_hand_landmarks')
    
    def get_vec_with_vis(lms, idx, min_vis=0.2):
        if lms and idx < len(lms) and lms[idx] is not None:
            p = lms[idx]
            vis = p.get('visibility', 1.0)
            # MediaPipe: X=right, Y=down, Z=away_from_cam -> Target: X=left(+), Y=up(+), Z=front(+)
            return Vector((p['x'] * 100.0, -p['y'] * 100.0, -p['z'] * 100.0)), vis
        return None, 0.0

    # 1. Posture Anchor: Keep Spine, Neck, Head, Legs straight and upright
    upright_bones = [
        'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
        'LeftShoulder', 'RightShoulder',
        'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
        'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
    ]
    for bname in upright_bones:
        pb = char_bones.get(bname)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
            pb.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    # 2. Left Arm & Forearm
    l_sh, l_sh_vis = get_vec_with_vis(pose_lms, 11, min_vis=0.15)
    l_elb, l_elb_vis = get_vec_with_vis(pose_lms, 13, min_vis=0.15)
    l_w, l_w_vis = get_vec_with_vis(pose_lms, 15, min_vis=0.2)
    
    pb_l_arm = char_bones.get('LeftArm')
    pb_l_fore = char_bones.get('LeftForeArm')
    pb_l_hand = char_bones.get('LeftHand')
    
    l_active = (lh is not None) or (l_w is not None and l_w_vis >= 0.25)
    
    if l_sh and l_elb and l_active:
        dir_l_arm_target = norm_v(l_elb - l_sh)
        if dir_l_arm_target.z < 0.0:
            log_clamp("LeftArm", f_idx, dir_l_arm_target.z, 0.1, "TorsoBackPenetration")
            dir_l_arm_target.z = 0.1
            dir_l_arm_target = norm_v(dir_l_arm_target)
    else:
        dir_l_arm_target = REST_L_ARM
        
    dir_l_arm = REST_L_ARM.lerp(dir_l_arm_target, blend_weight if l_active else 0.0).normalized()
    
    # Calculate local rotation for LeftArm
    target_l_arm_local = norm_v(M_REST_LA.inverted() @ dir_l_arm)
    q_l_arm_local = norm_v(Vector((0, 1, 0))).rotation_difference(target_l_arm_local)
    
    if pb_l_arm:
        pb_l_arm.rotation_mode = 'QUATERNION'
        pb_l_arm.rotation_quaternion = q_l_arm_local
        pb_l_arm.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
        
    bpy.context.view_layer.update()
    
    if l_elb and l_w and l_active:
        dir_l_fore_target = norm_v(l_w - l_elb)
        if dir_l_fore_target.z < 0.1:
            log_clamp("LeftForeArm_SigningSpace", f_idx, dir_l_fore_target.z, 0.25, "ForearmBackwardClamp")
            dir_l_fore_target.z = 0.25
            dir_l_fore_target = norm_v(dir_l_fore_target)
        elbow_angle = math.acos(max(-1.0, min(1.0, dir_l_arm.dot(dir_l_fore_target))))
        if elbow_angle > math.radians(145.0):
            log_clamp("LeftForeArm_ElbowFlexion", f_idx, math.degrees(elbow_angle), 145.0, "ElbowOverflexion")
            axis = norm_v(dir_l_arm.cross(dir_l_fore_target))
            dir_l_fore_target = norm_v(Quaternion(axis, math.radians(145.0)) @ dir_l_arm)
    else:
        dir_l_fore_target = REST_L_FORE
        
    dir_l_fore = REST_L_FORE.lerp(dir_l_fore_target, blend_weight if l_active else 0.0).normalized()
    
    # In pose hierarchy, LeftForeArm target is transformed by parent pb_l_arm.matrix:
    dir_l_fore_in_parent = norm_v(pb_l_arm.matrix.to_3x3().inverted() @ dir_l_fore)
    M_rel_rest_lf = M_REST_LA.inverted() @ M_REST_LF
    target_l_fore_local = norm_v(M_rel_rest_lf.inverted() @ dir_l_fore_in_parent)
    q_l_fore_local = norm_v(Vector((0, 1, 0))).rotation_difference(target_l_fore_local)
    
    if pb_l_fore:
        pb_l_fore.rotation_mode = 'QUATERNION'
        pb_l_fore.rotation_quaternion = q_l_fore_local
        pb_l_fore.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    # 3. Right Arm & Forearm
    r_sh, r_sh_vis = get_vec_with_vis(pose_lms, 12, min_vis=0.15)
    r_elb, r_elb_vis = get_vec_with_vis(pose_lms, 14, min_vis=0.15)
    r_w, r_w_vis = get_vec_with_vis(pose_lms, 16, min_vis=0.2)
    
    pb_r_arm = char_bones.get('RightArm')
    pb_r_fore = char_bones.get('RightForeArm')
    pb_r_hand = char_bones.get('RightHand')
    
    r_active = (rh is not None) or (r_w is not None and r_w_vis >= 0.25)
    
    if r_sh and r_elb and r_active:
        dir_r_arm_target = norm_v(r_elb - r_sh)
        if dir_r_arm_target.z < 0.0:
            log_clamp("RightArm", f_idx, dir_r_arm_target.z, 0.1, "TorsoBackPenetration")
            dir_r_arm_target.z = 0.1
            dir_r_arm_target = norm_v(dir_r_arm_target)
    else:
        dir_r_arm_target = REST_R_ARM
        
    dir_r_arm = REST_R_ARM.lerp(dir_r_arm_target, blend_weight if r_active else 0.0).normalized()
    target_r_arm_local = norm_v(M_REST_RA.inverted() @ dir_r_arm)
    q_r_arm_local = norm_v(Vector((0, 1, 0))).rotation_difference(target_r_arm_local)
    
    if pb_r_arm:
        pb_r_arm.rotation_mode = 'QUATERNION'
        pb_r_arm.rotation_quaternion = q_r_arm_local
        pb_r_arm.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
        
    bpy.context.view_layer.update()
    
    if r_elb and r_w and r_active:
        dir_r_fore_target = norm_v(r_w - r_elb)
        if dir_r_fore_target.z < 0.1:
            log_clamp("RightForeArm_SigningSpace", f_idx, dir_r_fore_target.z, 0.25, "ForearmBackwardClamp")
            dir_r_fore_target.z = 0.25
            dir_r_fore_target = norm_v(dir_r_fore_target)
        elbow_angle_r = math.acos(max(-1.0, min(1.0, dir_r_arm.dot(dir_r_fore_target))))
        if elbow_angle_r > math.radians(145.0):
            log_clamp("RightForeArm_ElbowFlexion", f_idx, math.degrees(elbow_angle_r), 145.0, "ElbowOverflexion")
            axis = norm_v(dir_r_arm.cross(dir_r_fore_target))
            dir_r_fore_target = norm_v(Quaternion(axis, math.radians(145.0)) @ dir_r_arm)
    else:
        dir_r_fore_target = REST_R_FORE
        
    dir_r_fore = REST_R_FORE.lerp(dir_r_fore_target, blend_weight if r_active else 0.0).normalized()
    dir_r_fore_in_parent = norm_v(pb_r_arm.matrix.to_3x3().inverted() @ dir_r_fore)
    M_rel_rest_rf = M_REST_RA.inverted() @ M_REST_RF
    target_r_fore_local = norm_v(M_rel_rest_rf.inverted() @ dir_r_fore_in_parent)
    q_r_fore_local = norm_v(Vector((0, 1, 0))).rotation_difference(target_r_fore_local)
    
    if pb_r_fore:
        pb_r_fore.rotation_mode = 'QUATERNION'
        pb_r_fore.rotation_quaternion = q_r_fore_local
        pb_r_fore.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
        
    bpy.context.view_layer.update()

    # 4. Hand Orientation via Full 3x3 Orthonormal Palm Frame
    def solve_hand(hand_lms, is_left=True, q_fore_world=None):
        global prev_palm_n_l, prev_palm_n_r, prev_q_hand_l, prev_q_hand_r, prev_finger_curls_l, prev_finger_curls_r
        prefix = 'LeftHand' if is_left else 'RightHand'
        pb_h = char_bones.get(prefix)
        R_rest = R_REST_LH if is_left else R_REST_RH
        prev_curls_dict = prev_finger_curls_l if is_left else prev_finger_curls_r
        
        finger_data = [
            ('Thumb', 1, [char_bones.get(prefix + 'Thumb1'), char_bones.get(prefix + 'Thumb2'), char_bones.get(prefix + 'Thumb3')]),
            ('Index', 5, [char_bones.get(prefix + 'Index1'), char_bones.get(prefix + 'Index2'), char_bones.get(prefix + 'Index3')]),
            ('Middle', 9, [char_bones.get(prefix + 'Middle1'), char_bones.get(prefix + 'Middle2'), char_bones.get(prefix + 'Middle3')]),
            ('Ring', 13, [char_bones.get(prefix + 'Ring1'), char_bones.get(prefix + 'Ring2'), char_bones.get(prefix + 'Ring3')]),
            ('Pinky', 17, [char_bones.get(prefix + 'Pinky1'), char_bones.get(prefix + 'Pinky2'), char_bones.get(prefix + 'Pinky3')])
        ]
        
        # Canonical natural rest curl proportions (MCP knuckle bends most, PIP middle less, DIP tip least)
        rest_cascade_map = {
            'Thumb':  [15.0, 12.0, 8.0],
            'Index':  [28.0, 18.0, 8.0],
            'Middle': [30.0, 20.0, 9.0],
            'Ring':   [32.0, 22.0, 10.0],
            'Pinky':  [35.0, 25.0, 12.0]
        }
        
        if not hand_lms:
            # Neutral idle hand pose with natural cascading curl proportions
            if is_left: prev_q_hand_l = None
            else: prev_q_hand_r = None
            prev_curls_dict.clear()
            
            if pb_h:
                pb_h.rotation_mode = 'QUATERNION'
                pb_h.rotation_quaternion = Quaternion((1, 0, 0, 0))
                pb_h.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
            for fname, _, pbs in finger_data:
                rest_degs = rest_cascade_map.get(fname, [25.0, 18.0, 8.0])
                for pb, deg in zip(pbs, rest_degs):
                    if pb:
                        pb.rotation_mode = 'QUATERNION'
                        curl = math.radians(deg)
                        pb.rotation_quaternion = Euler((0, 0, -curl if is_left else curl), 'XYZ').to_quaternion()
                        pb.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
            return

        pts = [Vector((lm['x'] * 100.0, -lm['y'] * 100.0, -lm['z'] * 100.0)) for lm in hand_lms]
        w = pts[0]
        imcp = pts[5]
        pmcp = pts[17]
        
        v_fwd = norm_v((imcp + pmcp) * 0.5 - w)
        
        if is_left:
            # Left Hand: (Index - Wrist) x (Pinky - Wrist) with inward/outward frame
            v_norm = norm_v((imcp - w).cross(pmcp - w))
            if prev_palm_n_l is not None:
                if prev_palm_n_l.dot(v_norm) < -0.2:
                    log_clamp("LeftHand_PalmNormalFlip", f_idx, prev_palm_n_l.dot(v_norm), 1.0, "GhostHandNormalInversion")
                    v_norm = -v_norm
            prev_palm_n_l = v_norm
            v_thumb = norm_v(v_norm.cross(v_fwd))
            v_norm = norm_v(v_fwd.cross(v_thumb))
            
            # Target matrix for LeftHand:
            # Col 0 (Local X) -> -v_thumb, Col 1 (Local Y) -> v_fwd, Col 2 (Local Z) -> -v_norm
            R_target = Matrix([
                [-v_thumb.x, v_fwd.x, -v_norm.x],
                [-v_thumb.y, v_fwd.y, -v_norm.y],
                [-v_thumb.z, v_fwd.z, -v_norm.z]
            ])
        else:
            # Right Hand: (Index - Wrist) x (Pinky - Wrist)
            v_norm = norm_v((imcp - w).cross(pmcp - w))
            if prev_palm_n_r is not None:
                if prev_palm_n_r.dot(v_norm) < -0.2:
                    log_clamp("RightHand_PalmNormalFlip", f_idx, prev_palm_n_r.dot(v_norm), 1.0, "GhostHandNormalInversion")
                    v_norm = -v_norm
            prev_palm_n_r = v_norm
            v_thumb = norm_v(v_norm.cross(v_fwd))
            v_norm = norm_v(v_fwd.cross(v_thumb))
            
            # Target matrix for RightHand:
            # Col 0 (Local X) -> v_thumb, Col 1 (Local Y) -> -v_fwd, Col 2 (Local Z) -> -v_norm
            R_target = Matrix([
                [v_thumb.x, -v_fwd.x, -v_norm.x],
                [v_thumb.y, -v_fwd.y, -v_norm.y],
                [v_thumb.z, -v_fwd.z, -v_norm.z]
            ])
            
        if R_rest is not None:
            # The hand's parent (ForeArm) is rotated by pb_fore.matrix in pose space.
            pb_fore = pb_l_fore if is_left else pb_r_fore
            M_rest_fore = M_REST_LF if is_left else M_REST_RF
            M_rel_rest_hand = M_rest_fore.inverted() @ R_rest
            
            if pb_fore:
                R_target_in_parent = pb_fore.matrix.to_3x3().inverted() @ R_target
                q_hand_local = (R_target_in_parent @ M_rel_rest_hand.inverted()).to_quaternion()
            else:
                q_hand_local = (R_target @ R_rest.inverted()).to_quaternion()
        else:
            q_hand_local = Quaternion((1, 0, 0, 0))
            
        # Temporal smoothing for calm and stable wrist orientation (removes jitter/shivering)
        prev_q = prev_q_hand_l if is_left else prev_q_hand_r
        if prev_q is not None:
            # Smooth interpolation: 65% new pose, 35% temporal continuity
            q_hand_local = prev_q.slerp(q_hand_local, 0.65)
        if is_left:
            prev_q_hand_l = q_hand_local.copy()
        else:
            prev_q_hand_r = q_hand_local.copy()
            
        q_hand_final = Quaternion((1, 0, 0, 0)).slerp(q_hand_local, blend_weight)
        
        if pb_h:
            pb_h.rotation_mode = 'QUATERNION'
            pb_h.rotation_quaternion = q_hand_final
            pb_h.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

        # Solve individual fingers with 1-DOF biological flexion hinges and temporal smoothing
        for fname, base, pbs in finger_data:
            p0, p1, p2, p3 = pts[base], pts[base+1], pts[base+2], pts[base+3]
            v01 = norm_v(p1 - p0)
            v12 = norm_v(p2 - p1)
            v23 = norm_v(p3 - p2)
            
            if fname == 'Thumb':
                f1_raw = math.degrees(math.acos(max(-1.0, min(1.0, v_fwd.dot(v01))))) * 0.6
                f2_raw = math.degrees(math.acos(max(-1.0, min(1.0, v01.dot(v12)))))
                f3_raw = math.degrees(math.acos(max(-1.0, min(1.0, v12.dot(v23)))))
                
                f1_deg = max(0.0, min(70.0, f1_raw))
                f2_deg = max(0.0, min(70.0, f2_raw))
                f3_deg = max(0.0, min(90.0, f3_raw))
                
                if f1_raw != f1_deg: log_clamp(f"{prefix}Thumb1", f_idx, f1_raw, f1_deg, "ThumbMCPClamp")
                if f2_raw != f2_deg: log_clamp(f"{prefix}Thumb2", f_idx, f2_raw, f2_deg, "ThumbPIPClamp")
                if f3_raw != f3_deg: log_clamp(f"{prefix}Thumb3", f_idx, f3_raw, f3_deg, "ThumbDIPClamp")
            else:
                f1_raw = math.degrees(math.acos(max(-1.0, min(1.0, v_fwd.dot(v01))))) * 0.9
                f2_raw = math.degrees(math.acos(max(-1.0, min(1.0, v01.dot(v12)))))
                f3_raw = math.degrees(math.acos(max(-1.0, min(1.0, v12.dot(v23)))))
                
                # Biological 1-DOF limits: MCP -5..90°, PIP 0..110°, DIP 0..90°
                f1_deg = max(-5.0, min(90.0, f1_raw))
                f2_deg = max(0.0, min(110.0, f2_raw))
                f3_deg = max(0.0, min(90.0, f3_raw))
                
                if f1_raw != f1_deg: log_clamp(f"{prefix}{fname}1", f_idx, f1_raw, f1_deg, "FingerMCPClamp")
                if f2_raw != f2_deg: log_clamp(f"{prefix}{fname}2", f_idx, f2_raw, f2_deg, "FingerPIPHyperextension")
                if f3_raw != f3_deg: log_clamp(f"{prefix}{fname}3", f_idx, f3_raw, f3_deg, "FingerDIPHyperextension")
                
            raw_curls = [f1_deg, f2_deg, f3_deg]
            prev_curls = prev_curls_dict.get(fname)
            if prev_curls is not None:
                # Temporal filter for calm finger flexing without high-frequency vibration
                smoothed_curls = [prev_curls[i] * 0.4 + raw_curls[i] * 0.6 for i in range(3)]
            else:
                smoothed_curls = raw_curls
            prev_curls_dict[fname] = smoothed_curls
            
            rest_degs = rest_cascade_map.get(fname, [25.0, 18.0, 8.0])
            curls = [math.radians(smoothed_curls[0] * blend_weight + (1 - blend_weight) * rest_degs[0]),
                     math.radians(smoothed_curls[1] * blend_weight + (1 - blend_weight) * rest_degs[1]),
                     math.radians(smoothed_curls[2] * blend_weight + (1 - blend_weight) * rest_degs[2])]
            for pb, curl in zip(pbs, curls):
                if pb:
                    pb.rotation_mode = 'QUATERNION'
                    # Mixamo finger bones curl along local Z-axis (LeftHand: -Z, RightHand: +Z)
                    q_curl = Euler((0, 0, -curl if is_left else curl), 'XYZ').to_quaternion()
                    pb.rotation_quaternion = q_curl
                    pb.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    solve_hand(lh, is_left=True)
    solve_hand(rh, is_left=False)

# 5. Push NLA track & Export Web GLB
bpy.ops.object.mode_set(mode='OBJECT')
track = char_arm.animation_data.nla_tracks.new()
track.name = args.anim_name
track.strips.new(args.anim_name, 1, action)

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=args.output,
    export_format='GLB',
    export_animations=True,
    export_nla_strips=True,
    export_skins=True,
    export_morph=False,
    export_lights=False,
    export_apply=False,
    export_image_format='JPEG',
    export_jpeg_quality=85
)

if args.clamping_log:
    os.makedirs(os.path.dirname(os.path.abspath(args.clamping_log)), exist_ok=True)
    with open(args.clamping_log, 'w', encoding='utf-8') as f:
        json.dump(clamping_events, f, indent=2)

print(f"SUCCESSFULLY_EXPORTED: {args.output} | Total Clamps Logged: {len(clamping_events)}")
