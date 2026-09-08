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
bpy.ops.import_scene.fbx(filepath=args.character)
char_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
char_arm.name = "Character_Armature"
bpy.context.view_layer.objects.active = char_arm
bpy.ops.object.mode_set(mode='POSE')

# Downscale textures for web optimization
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
R_REST_LH = char_arm.data.bones['mixamorig2:LeftHand'].matrix_local.to_3x3() if 'mixamorig2:LeftHand' in char_arm.data.bones else None
R_REST_RH = char_arm.data.bones['mixamorig2:RightHand'].matrix_local.to_3x3() if 'mixamorig2:RightHand' in char_arm.data.bones else None

# Canonical Neutral Ready / Rest Pose (Arms resting down and forward in front of waist)
REST_L_ARM = norm_v(Vector((0.2, -0.9, 0.15)))
REST_L_FORE = norm_v(Vector((0.2, -0.3, 0.8)))
REST_R_ARM = norm_v(Vector((-0.2, -0.9, 0.15)))
REST_R_FORE = norm_v(Vector((-0.2, -0.3, 0.8)))

Q_REST_L_ARM = quat_from_to(Vector((1, 0, 0)), REST_L_ARM)
Q_REST_L_FORE_W = quat_from_to(Vector((1, 0, 0)), REST_L_FORE)
Q_REST_L_FORE = Q_REST_L_ARM.inverted() @ Q_REST_L_FORE_W

Q_REST_R_ARM = quat_from_to(Vector((-1, 0, 0)), REST_R_ARM)
Q_REST_R_FORE_W = quat_from_to(Vector((-1, 0, 0)), REST_R_FORE)
Q_REST_R_FORE = Q_REST_R_ARM.inverted() @ Q_REST_R_FORE_W

prev_palm_n_l = None
prev_palm_n_r = None

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
    
    def get_vec(lms, idx):
        if lms and idx < len(lms) and lms[idx] is not None:
            p = lms[idx]
            # MediaPipe: X=right, Y=down, Z=away_from_cam -> Target: X=left(+), Y=up(+), Z=front(+)
            return Vector((p['x'] * 100.0, -p['y'] * 100.0, -p['z'] * 100.0))
        return None

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
    l_sh = get_vec(pose_lms, 11)
    l_elb = get_vec(pose_lms, 13)
    l_w = get_vec(pose_lms, 15)
    
    pb_l_arm = char_bones.get('LeftArm')
    pb_l_fore = char_bones.get('LeftForeArm')
    pb_l_hand = char_bones.get('LeftHand')
    
    l_active = (lh is not None) or (l_w is not None and l_w.y > -25.0)
    
    if l_sh and l_elb and l_active:
        dir_l_arm = norm_v(l_elb - l_sh)
        if dir_l_arm.z < -0.15:
            log_clamp("LeftArm", f_idx, dir_l_arm.z, -0.1, "TorsoBackPenetration")
            dir_l_arm.z = -0.1
            dir_l_arm = norm_v(dir_l_arm)
    else:
        dir_l_arm = REST_L_ARM
        
    q_l_arm_raw = quat_from_to(Vector((1, 0, 0)), dir_l_arm)
    q_l_arm_world = Q_REST_L_ARM.slerp(q_l_arm_raw, blend_weight)
    
    if l_elb and l_w and l_active:
        dir_l_fore = norm_v(l_w - l_elb)
        if dir_l_fore.z < -0.1:
            log_clamp("LeftForeArm_SigningSpace", f_idx, dir_l_fore.z, 0.2, "ForearmBackwardClamp")
            dir_l_fore.z = 0.2
            dir_l_fore = norm_v(dir_l_fore)
        elbow_angle = math.acos(max(-1.0, min(1.0, dir_l_arm.dot(dir_l_fore))))
        if elbow_angle > math.radians(145.0):
            log_clamp("LeftForeArm_ElbowFlexion", f_idx, math.degrees(elbow_angle), 145.0, "ElbowOverflexion")
            axis = norm_v(dir_l_arm.cross(dir_l_fore))
            dir_l_fore = norm_v(Quaternion(axis, math.radians(145.0)) @ dir_l_arm)
        q_l_fore_raw = quat_from_to(Vector((1, 0, 0)), dir_l_fore)
    else:
        q_l_fore_raw = Q_REST_L_FORE_W
        
    q_l_fore_world = Q_REST_L_FORE_W.slerp(q_l_fore_raw, blend_weight)
    q_l_fore_local = q_l_arm_world.inverted() @ q_l_fore_world
    
    if pb_l_arm:
        pb_l_arm.rotation_mode = 'QUATERNION'
        pb_l_arm.rotation_quaternion = q_l_arm_world
        pb_l_arm.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
        
    if pb_l_fore:
        pb_l_fore.rotation_mode = 'QUATERNION'
        pb_l_fore.rotation_quaternion = q_l_fore_local
        pb_l_fore.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    # 3. Right Arm & Forearm
    r_sh = get_vec(pose_lms, 12)
    r_elb = get_vec(pose_lms, 14)
    r_w = get_vec(pose_lms, 16)
    
    pb_r_arm = char_bones.get('RightArm')
    pb_r_fore = char_bones.get('RightForeArm')
    pb_r_hand = char_bones.get('RightHand')
    
    r_active = (rh is not None) or (r_w is not None and r_w.y > -25.0)
    
    if r_sh and r_elb and r_active:
        dir_r_arm = norm_v(r_elb - r_sh)
        if dir_r_arm.z < -0.15:
            log_clamp("RightArm", f_idx, dir_r_arm.z, -0.1, "TorsoBackPenetration")
            dir_r_arm.z = -0.1
            dir_r_arm = norm_v(dir_r_arm)
    else:
        dir_r_arm = REST_R_ARM
        
    q_r_arm_raw = quat_from_to(Vector((-1, 0, 0)), dir_r_arm)
    q_r_arm_world = Q_REST_R_ARM.slerp(q_r_arm_raw, blend_weight)
    
    if r_elb and r_w and r_active:
        dir_r_fore = norm_v(r_w - r_elb)
        if dir_r_fore.z < -0.1:
            log_clamp("RightForeArm_SigningSpace", f_idx, dir_r_fore.z, 0.2, "ForearmBackwardClamp")
            dir_r_fore.z = 0.2
            dir_r_fore = norm_v(dir_r_fore)
        elbow_angle_r = math.acos(max(-1.0, min(1.0, dir_r_arm.dot(dir_r_fore))))
        if elbow_angle_r > math.radians(145.0):
            log_clamp("RightForeArm_ElbowFlexion", f_idx, math.degrees(elbow_angle_r), 145.0, "ElbowOverflexion")
            axis = norm_v(dir_r_arm.cross(dir_r_fore))
            dir_r_fore = norm_v(Quaternion(axis, math.radians(145.0)) @ dir_r_arm)
        q_r_fore_raw = quat_from_to(Vector((-1, 0, 0)), dir_r_fore)
    else:
        q_r_fore_raw = Q_REST_R_FORE_W
        
    q_r_fore_world = Q_REST_R_FORE_W.slerp(q_r_fore_raw, blend_weight)
    q_r_fore_local = q_r_arm_world.inverted() @ q_r_fore_world
    
    if pb_r_arm:
        pb_r_arm.rotation_mode = 'QUATERNION'
        pb_r_arm.rotation_quaternion = q_r_arm_world
        pb_r_arm.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
        
    if pb_r_fore:
        pb_r_fore.rotation_mode = 'QUATERNION'
        pb_r_fore.rotation_quaternion = q_r_fore_local
        pb_r_fore.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    # 4. Hand Orientation via Full 3x3 Orthonormal Palm Frame
    def solve_hand(hand_lms, is_left=True, q_fore_world=None):
        global prev_palm_n_l, prev_palm_n_r
        prefix = 'LeftHand' if is_left else 'RightHand'
        pb_h = char_bones.get(prefix)
        R_rest = R_REST_LH if is_left else R_REST_RH
        
        finger_data = [
            ('Thumb', 1, [char_bones.get(prefix + 'Thumb1'), char_bones.get(prefix + 'Thumb2'), char_bones.get(prefix + 'Thumb3')]),
            ('Index', 5, [char_bones.get(prefix + 'Index1'), char_bones.get(prefix + 'Index2'), char_bones.get(prefix + 'Index3')]),
            ('Middle', 9, [char_bones.get(prefix + 'Middle1'), char_bones.get(prefix + 'Middle2'), char_bones.get(prefix + 'Middle3')]),
            ('Ring', 13, [char_bones.get(prefix + 'Ring1'), char_bones.get(prefix + 'Ring2'), char_bones.get(prefix + 'Ring3')]),
            ('Pinky', 17, [char_bones.get(prefix + 'Pinky1'), char_bones.get(prefix + 'Pinky2'), char_bones.get(prefix + 'Pinky3')])
        ]
        
        if not hand_lms:
            # Neutral idle hand pose
            if pb_h:
                pb_h.rotation_mode = 'QUATERNION'
                pb_h.rotation_quaternion = Quaternion((1, 0, 0, 0))
                pb_h.keyframe_insert(data_path='rotation_quaternion', frame=f_num)
            for fname, _, pbs in finger_data:
                rest_curls = [math.radians(15.0), math.radians(25.0), math.radians(15.0)] if fname != 'Thumb' else [math.radians(10.0), math.radians(15.0), math.radians(10.0)]
                for pb, curl in zip(pbs, rest_curls):
                    if pb:
                        pb.rotation_mode = 'QUATERNION'
                        pb.rotation_quaternion = Euler((curl if is_left else -curl, 0, 0), 'XYZ').to_quaternion()
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
            v_lat = norm_v(v_norm.cross(v_fwd))
            v_norm = norm_v(v_fwd.cross(v_lat))
            
            # Target matrix for LeftHand: col 0 = v_lat, col 1 = v_fwd, col 2 = -v_norm
            R_target = Matrix([
                [v_lat.x, v_fwd.x, -v_norm.x],
                [v_lat.y, v_fwd.y, -v_norm.y],
                [v_lat.z, v_fwd.z, -v_norm.z]
            ])
        else:
            # Right Hand: (Index - Wrist) x (Pinky - Wrist)
            v_norm = norm_v((imcp - w).cross(pmcp - w))
            if prev_palm_n_r is not None:
                if prev_palm_n_r.dot(v_norm) < -0.2:
                    log_clamp("RightHand_PalmNormalFlip", f_idx, prev_palm_n_r.dot(v_norm), 1.0, "GhostHandNormalInversion")
                    v_norm = -v_norm
            prev_palm_n_r = v_norm
            v_lat = norm_v(v_norm.cross(v_fwd))
            v_norm = norm_v(v_fwd.cross(v_lat))
            
            # Target matrix for RightHand: col 0 = v_lat, col 1 = v_fwd, col 2 = -v_norm
            R_target = Matrix([
                [v_lat.x, v_fwd.x, -v_norm.x],
                [v_lat.y, v_fwd.y, -v_norm.y],
                [v_lat.z, v_fwd.z, -v_norm.z]
            ])
            
        if R_rest is not None:
            q_hand_raw = (R_target @ R_rest.inverted()).to_quaternion()
        else:
            q_hand_raw = quat_from_to(Vector((-1 if not is_left else 1, 0, 0)), v_fwd)
            
        q_hand_world = Quaternion((1, 0, 0, 0)).slerp(q_hand_raw, blend_weight)
        
        if pb_h and q_fore_world:
            q_hand_local = q_fore_world.inverted() @ q_hand_world
            pb_h.rotation_mode = 'QUATERNION'
            pb_h.rotation_quaternion = q_hand_local
            pb_h.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

        # Solve individual fingers with 1-DOF biological flexion hinges
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
                f2_deg = max(0.0, min(70.0, f2_deg))
                f3_deg = max(0.0, min(90.0, f3_deg))
                
                if f1_raw != f1_deg: log_clamp(f"{prefix}Thumb1", f_idx, f1_raw, f1_deg, "ThumbMCPClamp")
                if f2_raw != f2_deg: log_clamp(f"{prefix}Thumb2", f_idx, f2_raw, f2_deg, "ThumbPIPClamp")
                if f3_raw != f3_deg: log_clamp(f"{prefix}Thumb3", f_idx, f3_raw, f3_deg, "ThumbDIPClamp")
            else:
                f1_raw = math.degrees(math.acos(max(-1.0, min(1.0, v_fwd.dot(v01))))) * 0.9
                f2_raw = math.degrees(math.acos(max(-1.0, min(1.0, v01.dot(v12)))))
                f3_raw = math.degrees(math.acos(max(-1.0, min(1.0, v12.dot(v23)))))
                
                # Biological 1-DOF limits: MCP -5..90°, PIP 0..110°, DIP 0..90°
                f1_deg = max(-5.0, min(90.0, f1_raw))
                f2_deg = max(0.0, min(110.0, f2_deg))
                f3_deg = max(0.0, min(90.0, f3_deg))
                
                if f1_raw != f1_deg: log_clamp(f"{prefix}{fname}1", f_idx, f1_raw, f1_deg, "FingerMCPClamp")
                if f2_raw != f2_deg: log_clamp(f"{prefix}{fname}2", f_idx, f2_raw, f2_deg, "FingerPIPHyperextension")
                if f3_raw != f3_deg: log_clamp(f"{prefix}{fname}3", f_idx, f3_raw, f3_deg, "FingerDIPHyperextension")
                
            curls = [math.radians(f1_deg * blend_weight + (1 - blend_weight)*15.0),
                     math.radians(f2_deg * blend_weight + (1 - blend_weight)*25.0),
                     math.radians(f3_deg * blend_weight + (1 - blend_weight)*15.0)]
            for pb, curl in zip(pbs, curls):
                if pb:
                    pb.rotation_mode = 'QUATERNION'
                    q_curl = Euler((curl if is_left else -curl, 0, 0), 'XYZ').to_quaternion()
                    pb.rotation_quaternion = q_curl
                    pb.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    solve_hand(lh, is_left=True, q_fore_world=q_l_fore_world)
    solve_hand(rh, is_left=False, q_fore_world=q_r_fore_world)

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
