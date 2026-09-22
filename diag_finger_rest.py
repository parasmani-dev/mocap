import bpy, sys, math
from mathutils import Quaternion, Euler, Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r'C:\Users\paras\Downloads\Ch22_nonPBR.fbx')
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')

print('=== FINGER BONE REST POSE ROTATIONS ===')
for pb in arm.pose.bones:
    n = pb.name
    if any(x in n for x in ['Index','Middle','Ring','Pinky','Thumb']):
        # bone direction in local parent space
        tail_local = pb.bone.tail_local
        head_local = pb.bone.head_local
        bone_vec = (tail_local - head_local).normalized()
        print(f'{n}: euler={[round(math.degrees(a),1) for a in pb.bone.matrix_local.to_euler()]} bone_vec={[round(x,3) for x in bone_vec]}')

print('=== END ===')
