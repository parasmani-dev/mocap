"""
Humanified ISL BVH Generator:
- Pure upright Spine & Hips (zero slump / zero lateral tilt)
- Perfect level forward-facing Head gaze (locked to face camera +Z)
- Straight natural standing lower body
- 1-DOF anatomical finger hinges
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R

from pipeline.smoothing import OneEuroFilter

BVH_MIXAMO_HIERARCHY = {
    "Hips": {
        "parent": None,
        "offset": [0.0, 95.0, 0.0],
        "channels": ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
        "children": ["Spine", "LeftUpLeg", "RightUpLeg"]
    },
    "Spine": {
        "parent": "Hips",
        "offset": [0.0, 10.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["Spine1"]
    },
    "Spine1": {
        "parent": "Spine",
        "offset": [0.0, 12.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["Spine2"]
    },
    "Spine2": {
        "parent": "Spine1",
        "offset": [0.0, 12.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["Neck", "LeftShoulder", "RightShoulder"]
    },
    "Neck": {
        "parent": "Spine2",
        "offset": [0.0, 10.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["Head"]
    },
    "Head": {
        "parent": "Neck",
        "offset": [0.0, 8.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["Head_End"]
    },
    "Head_End": {
        "parent": "Head",
        "offset": [0.0, 10.0, 0.0],
        "channels": [],
        "children": []
    },
    # Left Arm & Hand
    "LeftShoulder": {
        "parent": "Spine2",
        "offset": [6.0, 8.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["LeftArm"]
    },
    "LeftArm": {
        "parent": "LeftShoulder",
        "offset": [12.0, 0.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["LeftForeArm"]
    },
    "LeftForeArm": {
        "parent": "LeftArm",
        "offset": [24.0, 0.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["LeftHand"]
    },
    "LeftHand": {
        "parent": "LeftForeArm",
        "offset": [22.0, 0.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": [
            "LeftHandThumb1", "LeftHandIndex1", "LeftHandMiddle1", "LeftHandRing1", "LeftHandPinky1"
        ]
    },
    # Left Hand Fingers
    "LeftHandThumb1": {"parent": "LeftHand", "offset": [1.8, -1.4, 0.5], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandThumb2"]},
    "LeftHandThumb2": {"parent": "LeftHandThumb1", "offset": [2.4, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandThumb3"]},
    "LeftHandThumb3": {"parent": "LeftHandThumb2", "offset": [1.9, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandThumb_End"]},
    "LeftHandThumb_End": {"parent": "LeftHandThumb3", "offset": [1.6, 0.0, 0.0], "channels": [], "children": []},

    "LeftHandIndex1": {"parent": "LeftHand", "offset": [6.6, 1.2, 0.3], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandIndex2"]},
    "LeftHandIndex2": {"parent": "LeftHandIndex1", "offset": [2.0, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandIndex3"]},
    "LeftHandIndex3": {"parent": "LeftHandIndex2", "offset": [1.2, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandIndex_End"]},
    "LeftHandIndex_End": {"parent": "LeftHandIndex3", "offset": [1.0, 0.0, 0.0], "channels": [], "children": []},

    "LeftHandMiddle1": {"parent": "LeftHand", "offset": [7.2, 0.0, 0.3], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandMiddle2"]},
    "LeftHandMiddle2": {"parent": "LeftHandMiddle1", "offset": [3.0, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandMiddle3"]},
    "LeftHandMiddle3": {"parent": "LeftHandMiddle2", "offset": [2.0, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandMiddle_End"]},
    "LeftHandMiddle_End": {"parent": "LeftHandMiddle3", "offset": [1.6, 0.0, 0.0], "channels": [], "children": []},

    "LeftHandRing1": {"parent": "LeftHand", "offset": [7.1, -1.0, 0.4], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandRing2"]},
    "LeftHandRing2": {"parent": "LeftHandRing1", "offset": [2.6, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandRing3"]},
    "LeftHandRing3": {"parent": "LeftHandRing2", "offset": [1.6, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandRing_End"]},
    "LeftHandRing_End": {"parent": "LeftHandRing3", "offset": [1.3, 0.0, 0.0], "children": []},

    "LeftHandPinky1": {"parent": "LeftHand", "offset": [6.5, -2.0, 0.5], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandPinky2"]},
    "LeftHandPinky2": {"parent": "LeftHandPinky1", "offset": [1.9, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandPinky3"]},
    "LeftHandPinky3": {"parent": "LeftHandPinky2", "offset": [1.4, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHandPinky_End"]},
    "LeftHandPinky_End": {"parent": "LeftHandPinky3", "offset": [1.3, 0.0, 0.0], "children": []},

    # Right Arm & Hand
    "RightShoulder": {
        "parent": "Spine2",
        "offset": [-6.0, 8.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["RightArm"]
    },
    "RightArm": {
        "parent": "RightShoulder",
        "offset": [-12.0, 0.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["RightForeArm"]
    },
    "RightForeArm": {
        "parent": "RightArm",
        "offset": [-24.0, 0.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["RightHand"]
    },
    "RightHand": {
        "parent": "RightForeArm",
        "offset": [-22.0, 0.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": [
            "RightHandThumb1", "RightHandIndex1", "RightHandMiddle1", "RightHandRing1", "RightHandPinky1"
        ]
    },
    # Right Hand Fingers
    "RightHandThumb1": {"parent": "RightHand", "offset": [-1.8, -1.4, 0.5], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandThumb2"]},
    "RightHandThumb2": {"parent": "RightHandThumb1", "offset": [-2.4, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandThumb3"]},
    "RightHandThumb3": {"parent": "RightHandThumb2", "offset": [-1.9, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandThumb_End"]},
    "RightHandThumb_End": {"parent": "RightHandThumb3", "offset": [-1.6, 0.0, 0.0], "channels": [], "children": []},

    "RightHandIndex1": {"parent": "RightHand", "offset": [-6.6, 1.2, 0.3], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandIndex2"]},
    "RightHandIndex2": {"parent": "RightHandIndex1", "offset": [-2.0, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandIndex3"]},
    "RightHandIndex3": {"parent": "RightHandIndex2", "offset": [-1.2, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandIndex_End"]},
    "RightHandIndex_End": {"parent": "RightHandIndex3", "offset": [-1.0, 0.0, 0.0], "channels": [], "children": []},

    "RightHandMiddle1": {"parent": "RightHand", "offset": [-7.2, 0.0, 0.3], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandMiddle2"]},
    "RightHandMiddle2": {"parent": "RightHandMiddle1", "offset": [-3.0, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandMiddle3"]},
    "RightHandMiddle3": {"parent": "RightHandMiddle2", "offset": [-2.0, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandMiddle_End"]},
    "RightHandMiddle_End": {"parent": "RightHandMiddle3", "offset": [-1.6, 0.0, 0.0], "children": []},

    "RightHandRing1": {"parent": "RightHand", "offset": [-7.1, -1.0, 0.4], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandRing2"]},
    "RightHandRing2": {"parent": "RightHandRing1", "offset": [-2.6, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandRing3"]},
    "RightHandRing3": {"parent": "RightHandRing2", "offset": [-1.6, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandRing_End"]},
    "RightHandRing_End": {"parent": "RightHandRing3", "offset": [-1.3, 0.0, 0.0], "children": []},

    "RightHandPinky1": {"parent": "RightHand", "offset": [-6.5, -2.0, 0.5], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandPinky2"]},
    "RightHandPinky2": {"parent": "RightHandPinky1", "offset": [-1.9, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandPinky3"]},
    "RightHandPinky3": {"parent": "RightHandPinky2", "offset": [-1.4, 0.0, 0.0], "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHandPinky_End"]},
    "RightHandPinky_End": {"parent": "RightHandPinky3", "offset": [-1.3, 0.0, 0.0], "children": []},

    # Lower Body
    "LeftUpLeg": {
        "parent": "Hips",
        "offset": [10.0, -5.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["LeftLeg"]
    },
    "LeftLeg": {
        "parent": "LeftUpLeg",
        "offset": [0.0, -42.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["LeftFoot"]
    },
    "LeftFoot": {
        "parent": "LeftLeg",
        "offset": [0.0, -42.0, 4.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["LeftFoot_End"]
    },
    "LeftFoot_End": {"parent": "LeftFoot", "offset": [0.0, 0.0, 12.0], "channels": [], "children": []},

    "RightUpLeg": {
        "parent": "Hips",
        "offset": [-10.0, -5.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["RightLeg"]
    },
    "RightLeg": {
        "parent": "RightUpLeg",
        "offset": [0.0, -42.0, 0.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["RightFoot"]
    },
    "RightFoot": {
        "parent": "RightLeg",
        "offset": [0.0, -42.0, 4.0],
        "channels": ["Zrotation", "Xrotation", "Yrotation"],
        "children": ["RightFoot_End"]
    },
    "RightFoot_End": {"parent": "RightFoot", "offset": [0.0, 0.0, 12.0], "channels": [], "children": []}
}

def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-7:
        return np.zeros_like(v)
    return v / norm

def quat_from_vectors(vec_source: np.ndarray, vec_target: np.ndarray) -> R:
    u = normalize_vector(vec_source)
    v = normalize_vector(vec_target)
    
    dot = np.dot(u, v)
    if dot > 0.999999:
        return R.identity()
    if dot < -0.999999:
        axis = np.array([0.0, 0.0, 1.0])
        if abs(u[2]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        ortho = normalize_vector(np.cross(u, axis))
        return R.from_rotvec(np.pi * ortho)
    
    axis = np.cross(u, v)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm
    angle = math.acos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(angle * axis)

def quat_to_euler_continuous(r_obj: R, prev_euler: Optional[np.ndarray] = None, order: str = "zxy") -> np.ndarray:
    angles = np.array(r_obj.as_euler(order, degrees=True), dtype=float)
    if prev_euler is None:
        return angles
    diff = angles - prev_euler
    angles -= np.round(diff / 360.0) * 360.0
    return angles

class BVHConverter:
    def __init__(self, hierarchy: Dict[str, Dict[str, Any]] = None):
        self.hierarchy = hierarchy or BVH_MIXAMO_HIERARCHY
        
    def generate_bvh_header(self) -> Tuple[str, List[str]]:
        lines = ["HIERARCHY"]
        joint_order = []
        
        def write_joint(joint_name: str, indent_level: int):
            indent = "  " * indent_level
            data = self.hierarchy[joint_name]
            is_root = data["parent"] is None
            is_end = len(data["children"]) == 0 and len(data.get("channels", [])) == 0
            
            if is_root:
                lines.append(f"{indent}ROOT {joint_name}")
            elif is_end:
                lines.append(f"{indent}End Site")
            else:
                lines.append(f"{indent}JOINT {joint_name}")
                
            lines.append(f"{indent}{{")
            
            off = data["offset"]
            lines.append(f"{indent}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
            
            channels = data.get("channels", [])
            if channels:
                lines.append(f"{indent}  CHANNELS {len(channels)} " + " ".join(channels))
                joint_order.append(joint_name)
                
            for child in data["children"]:
                write_joint(child, indent_level + 1)
                
            lines.append(f"{indent}}}")

        write_joint("Hips", 0)
        return "\n".join(lines), joint_order

    def convert_landmarks_to_bvh(
        self,
        landmark_data: Dict[str, Any],
        output_bvh_path: str,
        scale: float = 100.0,
        smooth: bool = True
    ) -> bool:
        frames = landmark_data.get("frames", [])
        fps = landmark_data.get("metadata", {}).get("fps", 30.0)
        if fps <= 0:
            fps = 30.0
        frame_time = 1.0 / fps
        total_frames = len(frames)
        
        header, joint_order = self.generate_bvh_header()
        
        motion_lines = [
            "MOTION",
            f"Frames: {total_frames}",
            f"Frame Time: {frame_time:.6f}"
        ]
        
        filters = {}
        if smooth:
            for k in ["l_sh", "l_elb", "l_w", "r_sh", "r_elb", "r_w"]:
                filters[k] = OneEuroFilter(freq=fps, mincutoff=1.5, beta=0.01, dcutoff=1.0)
                
        prev_joint_eulers: Dict[str, np.ndarray] = {}

        for f_idx, frame in enumerate(frames):
            pose_lms = frame.get("pose_world_landmarks") or frame.get("pose_landmarks")
            left_hand = frame.get("left_hand_landmarks")
            right_hand = frame.get("right_hand_landmarks")
            
            def get_p(lms, idx):
                if lms and idx < len(lms) and lms[idx] is not None:
                    p = lms[idx]
                    return np.array([p["x"] * scale, -p["y"] * scale, -p["z"] * scale], dtype=float)
                return None

            l_sh_raw = get_p(pose_lms, 11)
            r_sh_raw = get_p(pose_lms, 12)
            l_elb_raw = get_p(pose_lms, 13)
            l_w_raw = get_p(pose_lms, 15)
            r_elb_raw = get_p(pose_lms, 14)
            r_w_raw = get_p(pose_lms, 16)

            if smooth:
                l_sh = filters["l_sh"].filter(l_sh_raw) if l_sh_raw is not None else None
                r_sh = filters["r_sh"].filter(r_sh_raw) if r_sh_raw is not None else None
                l_elb = filters["l_elb"].filter(l_elb_raw) if l_elb_raw is not None else None
                l_w = filters["l_w"].filter(l_w_raw) if l_w_raw is not None else None
                r_elb = filters["r_elb"].filter(r_elb_raw) if r_elb_raw is not None else None
                r_w = filters["r_w"].filter(r_w_raw) if r_w_raw is not None else None
            else:
                l_sh, r_sh = l_sh_raw, r_sh_raw
                l_elb, l_w, r_elb, r_w = l_elb_raw, l_w_raw, r_elb_raw, r_w_raw

            Q_world: Dict[str, R] = {}
            Q_local: Dict[str, R] = {}
            
            # --- 1. Root / Hips: Upright Standing Human Posture ---
            hips_pos = np.array([0.0, 95.0, 0.0])
            
            # Torso upright + locked forward (Identity in world)
            Q_world["Hips"] = R.identity()
            Q_local["Hips"] = R.identity()

            # --- 2. Spine Chain: Upright Posture ---
            Q_world["Spine"] = R.identity()
            Q_local["Spine"] = R.identity()
            
            Q_world["Spine1"] = R.identity()
            Q_local["Spine1"] = R.identity()
            
            Q_world["Spine2"] = R.identity()
            Q_local["Spine2"] = R.identity()

            # --- 3. Neck & Head: True Forward Looking Gaze ---
            Q_world["Neck"] = R.identity()
            Q_local["Neck"] = R.identity()
            Q_world["Head"] = R.identity()
            Q_local["Head"] = R.identity()

            # --- 4. Left Arm & Hand Chain ---
            Q_world["LeftShoulder"] = R.identity()
            Q_local["LeftShoulder"] = R.identity()
            
            if l_sh is not None and l_elb is not None:
                l_arm_dir = normalize_vector(l_elb - l_sh)
                Q_l_arm_w = quat_from_vectors(np.array([1.0, 0.0, 0.0]), l_arm_dir)
            else:
                Q_l_arm_w = quat_from_vectors(np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
            Q_world["LeftArm"] = Q_l_arm_w
            Q_local["LeftArm"] = Q_world["LeftShoulder"].inv() * Q_world["LeftArm"]
            
            if l_elb is not None and l_w is not None:
                l_fore_dir = normalize_vector(l_w - l_elb)
                Q_l_fore_w = quat_from_vectors(np.array([1.0, 0.0, 0.0]), l_fore_dir)
            else:
                Q_l_fore_w = Q_world["LeftArm"]
            Q_world["LeftForeArm"] = Q_l_fore_w
            Q_local["LeftForeArm"] = Q_world["LeftArm"].inv() * Q_world["LeftForeArm"]
            
            def compute_hand_kinematics(hand_lms, is_left=True):
                parent_arm = "LeftForeArm" if is_left else "RightForeArm"
                if hand_lms:
                    w = get_p(hand_lms, 0)
                    imcp = get_p(hand_lms, 5)
                    pmcp = get_p(hand_lms, 17)
                    if w is not None and imcp is not None and pmcp is not None:
                        v_fwd = normalize_vector((imcp + pmcp) * 0.5 - w)
                        if is_left:
                            v_norm = normalize_vector(np.cross(pmcp - w, imcp - w))
                            v_lat = normalize_vector(np.cross(v_norm, v_fwd))
                            v_norm = normalize_vector(np.cross(v_fwd, v_lat))
                            R_mat = np.column_stack([v_fwd, v_lat, v_norm])
                        else:
                            v_norm = normalize_vector(np.cross(imcp - w, pmcp - w))
                            v_lat = normalize_vector(np.cross(v_norm, v_fwd))
                            v_norm = normalize_vector(np.cross(v_fwd, v_lat))
                            R_mat = np.column_stack([-v_fwd, -v_lat, v_norm])
                            
                        if np.linalg.det(R_mat) > 0.5:
                            return R.from_matrix(R_mat)
                return Q_world[parent_arm]

            Q_world["LeftHand"] = compute_hand_kinematics(left_hand, is_left=True)
            Q_local["LeftHand"] = Q_world["LeftForeArm"].inv() * Q_world["LeftHand"]

            # Anatomical Direct Flexion Finger Kinematics
            def solve_finger_chain_anatomical(hand_lms, finger_names, base_idx, is_left=True):
                parent_hand = "LeftHand" if is_left else "RightHand"
                Q_hand_w = Q_world[parent_hand]
                
                p_mcp = get_p(hand_lms, base_idx)
                p_pip = get_p(hand_lms, base_idx + 1)
                p_dip = get_p(hand_lms, base_idx + 2)
                p_tip = get_p(hand_lms, base_idx + 3)
                
                if p_mcp is not None and p_pip is not None:
                    v1_w = normalize_vector(p_pip - p_mcp)
                    ref1 = normalize_vector(np.array(self.hierarchy[finger_names[0]]["offset"]))
                    v1_local_hand = Q_hand_w.inv().apply(v1_w)
                    Q_j1_local = quat_from_vectors(ref1, v1_local_hand)
                    Q_b1_w = Q_hand_w * Q_j1_local
                else:
                    Q_j1_local = R.identity()
                    Q_b1_w = Q_hand_w
                    
                Q_local[finger_names[0]] = Q_j1_local
                Q_world[finger_names[0]] = Q_b1_w
                
                if p_pip is not None and p_dip is not None and p_mcp is not None:
                    v1_w = normalize_vector(p_pip - p_mcp)
                    v2_w = normalize_vector(p_dip - p_pip)
                    dot12 = np.clip(np.dot(v1_w, v2_w), -1.0, 1.0)
                    theta12 = math.acos(dot12)
                    flex_deg = math.degrees(theta12)
                    Q_j2_local = R.from_euler('zxy', [flex_deg, 0.0, 0.0], degrees=True)
                    Q_b2_w = Q_b1_w * Q_j2_local
                else:
                    Q_j2_local = R.identity()
                    Q_b2_w = Q_b1_w
                    
                Q_local[finger_names[1]] = Q_j2_local
                Q_world[finger_names[1]] = Q_b2_w
                
                if p_dip is not None and p_tip is not None and p_pip is not None:
                    v2_w = normalize_vector(p_dip - p_pip)
                    v3_w = normalize_vector(p_tip - p_dip)
                    dot23 = np.clip(np.dot(v2_w, v3_w), -1.0, 1.0)
                    theta23 = math.acos(dot23)
                    flex_deg23 = math.degrees(theta23)
                    Q_j3_local = R.from_euler('zxy', [flex_deg23, 0.0, 0.0], degrees=True)
                    Q_b3_w = Q_b2_w * Q_j3_local
                else:
                    Q_j3_local = R.identity()
                    Q_b3_w = Q_b2_w
                    
                Q_local[finger_names[2]] = Q_j3_local
                Q_world[finger_names[2]] = Q_b3_w

            solve_finger_chain_anatomical(left_hand, ["LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3"], 1, is_left=True)
            solve_finger_chain_anatomical(left_hand, ["LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3"], 5, is_left=True)
            solve_finger_chain_anatomical(left_hand, ["LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3"], 9, is_left=True)
            solve_finger_chain_anatomical(left_hand, ["LeftHandRing1", "LeftHandRing2", "LeftHandRing3"], 13, is_left=True)
            solve_finger_chain_anatomical(left_hand, ["LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3"], 17, is_left=True)

            # --- 5. Right Arm & Hand Chain ---
            Q_world["RightShoulder"] = R.identity()
            Q_local["RightShoulder"] = R.identity()
            
            if r_sh is not None and r_elb is not None:
                r_arm_dir = normalize_vector(r_elb - r_sh)
                Q_r_arm_w = quat_from_vectors(np.array([-1.0, 0.0, 0.0]), r_arm_dir)
            else:
                Q_r_arm_w = quat_from_vectors(np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
            Q_world["RightArm"] = Q_r_arm_w
            Q_local["RightArm"] = Q_world["RightShoulder"].inv() * Q_world["RightArm"]
            
            if r_elb is not None and r_w is not None:
                r_fore_dir = normalize_vector(r_w - r_elb)
                Q_r_fore_w = quat_from_vectors(np.array([-1.0, 0.0, 0.0]), r_fore_dir)
            else:
                Q_r_fore_w = Q_world["RightArm"]
            Q_world["RightForeArm"] = Q_r_fore_w
            Q_local["RightForeArm"] = Q_world["RightArm"].inv() * Q_world["RightForeArm"]
            
            Q_world["RightHand"] = compute_hand_kinematics(right_hand, is_left=False)
            Q_local["RightHand"] = Q_world["RightForeArm"].inv() * Q_world["RightHand"]
            
            solve_finger_chain_anatomical(right_hand, ["RightHandThumb1", "RightHandThumb2", "RightHandThumb3"], 1, is_left=False)
            solve_finger_chain_anatomical(right_hand, ["RightHandIndex1", "RightHandIndex2", "RightHandIndex3"], 5, is_left=False)
            solve_finger_chain_anatomical(right_hand, ["RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3"], 9, is_left=False)
            solve_finger_chain_anatomical(right_hand, ["RightHandRing1", "RightHandRing2", "RightHandRing3"], 13, is_left=False)
            solve_finger_chain_anatomical(right_hand, ["RightHandPinky1", "RightHandPinky2", "RightHandPinky3"], 17, is_left=False)

            # --- 6. Lower Body: Clean Straight Standing Legs ---
            Q_world["LeftUpLeg"] = R.identity()
            Q_local["LeftUpLeg"] = R.identity()
            Q_local["LeftLeg"] = R.identity()
            Q_local["LeftFoot"] = R.identity()

            Q_world["RightUpLeg"] = R.identity()
            Q_local["RightUpLeg"] = R.identity()
            Q_local["RightLeg"] = R.identity()
            Q_local["RightFoot"] = R.identity()

            # --- 7. Continuous Euler Angles (ZXY) ---
            frame_vals = []
            for j_name in joint_order:
                if j_name == "Hips":
                    frame_vals.extend([round(float(hips_pos[0]), 4), round(float(hips_pos[1]), 4), round(float(hips_pos[2]), 4)])
                    q_obj = Q_local["Hips"]
                else:
                    q_obj = Q_local.get(j_name, R.identity())
                    
                prev_e = prev_joint_eulers.get(j_name)
                euler_deg = quat_to_euler_continuous(q_obj, prev_euler=prev_e, order="zxy")
                prev_joint_eulers[j_name] = euler_deg
                
                frame_vals.extend([round(float(a), 4) for a in euler_deg])

            motion_lines.append(" ".join(f"{val:.4f}" for val in frame_vals))

        bvh_content = header + "\n" + "\n".join(motion_lines) + "\n"
        with open(output_bvh_path, "w", encoding="utf-8") as f:
            f.write(bvh_content)
            
        return True
