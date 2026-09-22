import bpy, sys, math
from mathutils import Quaternion, Euler, Vector, Matrix

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r'C:\Users\paras\Downloads\Ch22_nonPBR.fbx')
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
bpy.context.view_layer.objects.active = arm

# Get actual local axes for key finger bones
print('=== LOCAL AXES IN ARMATURE SPACE ===')
for bname in ['mixamorig2:RightHandIndex1','mixamorig2:RightHandMiddle1','mixamorig2:LeftHandIndex1']:
    if bname in arm.data.bones:
        b = arm.data.bones[bname]
        m = b.matrix_local.to_3x3()
        # Columns are local X, Y, Z axes in armature space
        lx = m.col[0]  # local X
        ly = m.col[1]  # local Y (bone direction)
        lz = m.col[2]  # local Z (typically curl axis)
        print(f'{bname}:')
        print(f'  local_Y (bone dir) = ({ly.x:.3f}, {ly.y:.3f}, {ly.z:.3f})')
        print(f'  local_X           = ({lx.x:.3f}, {lx.y:.3f}, {lx.z:.3f})')
        print(f'  local_Z           = ({lz.x:.3f}, {lz.y:.3f}, {lz.z:.3f})')

print('=== END ===')
