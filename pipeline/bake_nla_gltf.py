
import bpy
import sys
import argparse
import os

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anim_name", default="sign_action")
    return parser.parse_args(argv)

args = parse_args()
bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Import Character FBX
bpy.ops.import_scene.fbx(filepath=args.character)
char_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
char_arm.name = "Character_Armature"

# Downscale textures
for img in bpy.data.images:
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.scale(min(1024, img.size[0]), min(1024, img.size[1]))

# 2. Import BVH
bpy.ops.import_anim.bvh(filepath=args.bvh, target='ARMATURE', global_scale=1.0, update_scene_fps=True, update_scene_duration=True)
bvh_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o != char_arm][0]

bvh_action = bvh_arm.animation_data.action if bvh_arm.animation_data else None
frame_start = int(bvh_action.frame_range[0]) if bvh_action else 1
frame_end = int(bvh_action.frame_range[1]) if bvh_action else 60

# 3. Bone Constraints
char_bones = {}
for pb in char_arm.pose.bones:
    clean = pb.name
    for pfx in ["mixamorig2:", "mixamorig:", "mixamorig_"]:
        if clean.startswith(pfx):
            clean = clean[len(pfx):]
            break
    char_bones[clean.lower()] = pb

for bvh_pb in bvh_arm.pose.bones:
    clean_bvh = bvh_pb.name.lower()
    if clean_bvh in char_bones:
        target_pb = char_bones[clean_bvh]
        con = target_pb.constraints.new('COPY_ROTATION')
        con.target = bvh_arm
        con.subtarget = bvh_pb.name
        con.owner_space = 'WORLD'
        con.target_space = 'WORLD'

# 4. Bake Pose
bpy.context.view_layer.objects.active = char_arm
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.select_all(action='SELECT')
bpy.ops.nla.bake(
    frame_start=frame_start,
    frame_end=frame_end,
    only_selected=True,
    visual_keying=True,
    clear_constraints=True,
    bake_types={'POSE'}
)
bpy.ops.object.mode_set(mode='OBJECT')

baked_act = char_arm.animation_data.action
if baked_act:
    baked_act.name = args.anim_name
    # Push down into NLA track so glTF exporter is guaranteed to export it
    track = char_arm.animation_data.nla_tracks.new()
    track.name = args.anim_name
    track.strips.new(args.anim_name, frame_start, baked_act)

# Remove BVH
bpy.data.objects.remove(bvh_arm, do_unlink=True)
if bvh_action:
    bpy.data.actions.remove(bvh_action)

# 5. Export Web GLB
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
print("SUCCESS_EXPORT_GLB")
