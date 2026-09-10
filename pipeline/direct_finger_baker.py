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
    return parser.parse_args(argv)

args = parse_args()
bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Import Character FBX
bpy.ops.import_scene.fbx(filepath=args.character)
char_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
char_arm.name = "Character_Armature"
bpy.context.view_layer.objects.active = char_arm
bpy.ops.object.mode_set(mode='POSE')

# Downscale textures to 1024 for 4MB GLBs
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

with open(args.landmarks, 'r') as f:
    data = json.load(f)

frames = data.get('frames', [])
total_frames = len(frames)
if total_frames == 0:
    sys.exit(0)

fps = data.get('fps', 30.0)
if fps <= 0 or math.isnan(fps):
    fps = 30.0

if not char_arm.animation_data:
    char_arm.animation_data_create()
action = bpy.data.actions.new(name=args.anim_name)
char_arm.animation_data.action = action

def norm_v(v):
    l = v.length
    return v / l if l > 1e-7 else Vector((0,0,0))

def quat_between(u, v):
    u = norm_v(u)
    v = norm_v(v)
    dot = max(-1.0, min(1.0, u.dot(v)))
    if dot > 0.999999:
        return Quaternion((1,0,0,0))
    if dot < -0.999999:
        axis = Vector((0,0,1))
        if abs(u.z) > 0.9:
            axis = Vector((0,1,0))
        ortho = norm_v(u.cross(axis))
        return Quaternion(ortho, math.pi)
    axis = norm_v(u.cross(v))
    angle = math.acos(dot)
    return Quaternion(axis, angle)

def clamp_rad(val, min_deg, max_deg):
    min_r = math.radians(min_deg)
    max_r = math.radians(max_deg)
    return max(min_r, min(max_r, val))

scene = bpy.context.scene
scene.render.fps = int(round(fps))

prev_palm_normal_r = None
prev_palm_normal_l = None

for f_idx, frame in enumerate(frames):
    f_num = f_idx + 1
    scene.frame_set(f_num)
    
    pose_lms = frame.get('pose_world_landmarks') or frame.get('pose_landmarks')
    lh = frame.get('left_hand_landmarks')
    rh = frame.get('right_hand_landmarks')
    
    def get_vec(lms, idx):
        if lms and idx < len(lms) and lms[idx] is not None:
            p = lms[idx]
            return Vector((p['x'] * 100.0, -p['y'] * 100.0, -p['z'] * 100.0))
        return None

    # Upright Posture Locking
    for bname in ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot']:
        pb = char_bones.get(bname)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = Quaternion((1,0,0,0))
            pb.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    # Natural Signing Rest Pose directions
    REST_ARM_R = norm_v(Vector((-0.3, -0.9, 0.2)))
    REST_FORE_R = norm_v(Vector((-0.2, -0.8, 0.5)))
    REST_ARM_L = norm_v(Vector((0.3, -0.9, 0.2)))
    REST_FORE_L = norm_v(Vector((0.2, -0.8, 0.5)))

    # 1. Right Arm & ForeArm
    r_sh = get_vec(pose_lms, 12)
    r_elb = get_vec(pose_lms, 14)
    r_w = get_vec(pose_lms, 16)
    
    pb_r_arm = char_bones.get('RightArm')
    pb_r_fore = char_bones.get('RightForeArm')
    
    # Active detection: wrist in signing volume
    r_active = (rh is not None) or (r_w is not None and r_w.y > -0.2 * 100.0)
    
    if r_sh and r_elb and r_active:
        dir_arm = norm_v(r_elb - r_sh)
        if dir_arm.z < -0.3:  # Prevent arm from bending backwards
            dir_arm.z = -0.1
            dir_arm = norm_v(dir_arm)
    else:
        dir_arm = REST_ARM_R

    q_r_arm = quat_between(Vector((-1, 0, 0)), dir_arm)
    if pb_r_arm:
        pb_r_arm.rotation_mode = 'QUATERNION'
        pb_r_arm.rotation_quaternion = q_r_arm
        pb_r_arm.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    if r_elb and r_w and r_sh and r_active:
        dir_fore = norm_v(r_w - r_elb)
        if dir_fore.z < -0.2:  # Keep forearms forward in front of the body
            dir_fore.z = 0.1
            dir_fore = norm_v(dir_fore)
        elbow_angle = math.acos(max(-1.0, min(1.0, dir_arm.dot(dir_fore))))
        elbow_angle_clamped = clamp_rad(elbow_angle, 0.0, 145.0)
        q_r_fore_w = quat_between(Vector((-1, 0, 0)), dir_fore)
        q_r_fore_local = q_r_arm.inverted() @ q_r_fore_w
    else:
        q_r_fore_w = quat_between(Vector((-1, 0, 0)), REST_FORE_R)
        q_r_fore_local = q_r_arm.inverted() @ q_r_fore_w

    if pb_r_fore:
        pb_r_fore.rotation_mode = 'QUATERNION'
        pb_r_fore.rotation_quaternion = q_r_fore_local
        pb_r_fore.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    # 2. Left Arm & ForeArm
    l_sh = get_vec(pose_lms, 11)
    l_elb = get_vec(pose_lms, 13)
    l_w = get_vec(pose_lms, 15)
    
    pb_l_arm = char_bones.get('LeftArm')
    pb_l_fore = char_bones.get('LeftForeArm')
    
    l_active = (lh is not None) or (l_w is not None and l_w.y > -0.2 * 100.0)
    
    if l_sh and l_elb and l_active:
        dir_arm_l = norm_v(l_elb - l_sh)
        if dir_arm_l.z < -0.3:
            dir_arm_l.z = -0.1
            dir_arm_l = norm_v(dir_arm_l)
    else:
        dir_arm_l = REST_ARM_L

    q_l_arm = quat_between(Vector((1, 0, 0)), dir_arm_l)
    if pb_l_arm:
        pb_l_arm.rotation_mode = 'QUATERNION'
        pb_l_arm.rotation_quaternion = q_l_arm
        pb_l_arm.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    if l_elb and l_w and l_sh and l_active:
        dir_fore_l = norm_v(l_w - l_elb)
        if dir_fore_l.z < -0.2:
            dir_fore_l.z = 0.1
            dir_fore_l = norm_v(dir_fore_l)
        elbow_angle_l = math.acos(max(-1.0, min(1.0, dir_arm_l.dot(dir_fore_l))))
        elbow_angle_l_clamped = clamp_rad(elbow_angle_l, 0.0, 145.0)
        q_l_fore_w = quat_between(Vector((1, 0, 0)), dir_fore_l)
        q_l_fore_local = q_l_arm.inverted() @ q_l_fore_w
    else:
        q_l_fore_w = quat_between(Vector((1, 0, 0)), REST_FORE_L)
        q_l_fore_local = q_l_arm.inverted() @ q_l_fore_w

    if pb_l_fore:
        pb_l_fore.rotation_mode = 'QUATERNION'
        pb_l_fore.rotation_quaternion = q_l_fore_local
        pb_l_fore.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

    # 3. Direct Anatomical Finger Tracking with Palm-Normal Continuity & Neutral Rest Pose
    def solve_hand_fingers(hand_lms, is_left=False):
        global prev_palm_normal_r, prev_palm_normal_l
        prefix = 'LeftHand' if is_left else 'RightHand'
        pb_hand = char_bones.get(prefix)
        if pb_hand:
            pb_hand.rotation_mode = 'QUATERNION'
            pb_hand.rotation_quaternion = Quaternion((1,0,0,0))
            pb_hand.keyframe_insert(data_path='rotation_quaternion', frame=f_num)

        finger_defs = [
            ('Thumb', 1, [char_bones.get(prefix + 'Thumb1'), char_bones.get(prefix + 'Thumb2'), char_bones.get(prefix + 'Thumb3')]),
            ('Index', 5, [char_bones.get(prefix + 'Index1'), char_bones.get(prefix + 'Index2'), char_bones.get(prefix + 'Index3')]),
            ('Middle', 9, [char_bones.get(prefix + 'Middle1'), char_bones.get(prefix + 'Middle2'), char_bones.get(prefix + 'Middle3')]),
            ('Ring', 13, [char_bones.get(prefix + 'Ring1'), char_bones.get(prefix + 'Ring2'), char_bones.get(prefix + 'Ring3')]),
            ('Pinky', 17, [char_bones.get(prefix + 'Pinky1'), char_bones.get(prefix + 'Pinky2'), char_bones.get(prefix + 'Pinky3')])
        ]

        if not hand_lms:
            # Straight neutral rest pose when hand is not detected — fingers fully extended (0°)
            for fname, _, pbs in finger_defs:
                for pb in pbs:
                    if pb:
                        pb.rotation_mode = 'XYZ'
                        pb.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
                        pb.keyframe_insert(data_path='rotation_euler', frame=f_num)
            return

        pts = [Vector((lm['x'], lm['y'], lm['z'])) for lm in hand_lms]
        w = pts[0]
        imcp = pts[5]
        pmcp = pts[17]
        v_fwd = norm_v((imcp + pmcp)*0.5 - w)

        # Full Orthonormal Palm Normal with Temporal Continuity
        if is_left:
            v_norm = norm_v((pmcp - w).cross(imcp - w))
            if prev_palm_normal_l is not None:
                if prev_palm_normal_l.dot(v_norm) < -0.2:
                    v_norm = -v_norm # Correct ghost hand flip
            prev_palm_normal_l = v_norm
        else:
            v_norm = norm_v((imcp - w).cross(pmcp - w))
            if prev_palm_normal_r is not None:
                if prev_palm_normal_r.dot(v_norm) < -0.2:
                    v_norm = -v_norm # Correct ghost hand flip
            prev_palm_normal_r = v_norm

        for fname, base, pbs in finger_defs:
            p0, p1, p2, p3 = pts[base], pts[base+1], pts[base+2], pts[base+3]
            v01 = norm_v(p1 - p0)
            v12 = norm_v(p2 - p1)
            v23 = norm_v(p3 - p2)

            # Dead-zone: angles below 12° are treated as zero (finger is straight)
            # This prevents MediaPipe landmark noise from causing phantom curl
            DEAD_ZONE = math.radians(12.0)

            def apply_dead_zone(angle_rad):
                return max(0.0, angle_rad - DEAD_ZONE)

            if fname == 'Thumb':
                f1_raw = apply_dead_zone(math.acos(max(-1.0, min(1.0, v_fwd.dot(v01)))) * 0.5)
                f2_raw = apply_dead_zone(math.acos(max(-1.0, min(1.0, v01.dot(v12)))))
                f3_raw = apply_dead_zone(math.acos(max(-1.0, min(1.0, v12.dot(v23)))))

                f1 = clamp_rad(f1_raw, 0.0, 70.0)
                f2 = clamp_rad(f2_raw, 0.0, 70.0)
                f3 = clamp_rad(f3_raw, 0.0, 90.0)
            else:
                # MCP: measured against palm-forward direction; scale by 0.7 to reduce over-bend
                f1_raw = apply_dead_zone(math.acos(max(-1.0, min(1.0, v_fwd.dot(v01)))) * 0.7)
                f2_raw = apply_dead_zone(math.acos(max(-1.0, min(1.0, v01.dot(v12)))))
                f3_raw = apply_dead_zone(math.acos(max(-1.0, min(1.0, v12.dot(v23)))))

                # Clamping to biological 1-DOF hinge limits (no hyperextension/backward bend)
                f1 = clamp_rad(f1_raw, 0.0, 90.0)   # MCP: never allow negative (backward) curl
                f2 = clamp_rad(f2_raw, 0.0, 110.0)
                f3 = clamp_rad(f3_raw, 0.0, 90.0)

            for pb, angle in zip(pbs, [f1, f2, f3]):
                if pb:
                    pb.rotation_mode = 'XYZ'
                    pb.rotation_euler = Euler((angle, 0, 0), 'XYZ')
                    pb.keyframe_insert(data_path='rotation_euler', frame=f_num)

    solve_hand_fingers(rh, is_left=False)
    solve_hand_fingers(lh, is_left=True)

# 4. Push NLA track & Export Web GLB
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
print("SUCCESSFULLY_EXPORTED:", args.output)
