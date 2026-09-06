"""Configuration parameters, skeleton definitions, biological joint limits, and QA thresholds for ISL Mocap Pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

# Global Anatomical Joint Rotation Limits in Degrees [min_angle, max_angle]
# Enforces human kinematics and strictly prevents ghost flips, hyperextension, or joint twisting.
ANATOMICAL_JOINT_LIMITS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Elbow: Pure 1-DOF hinge flexion (0° straight to 145° fully bent)
    "LeftForeArm": {
        "flexion": (0.0, 145.0),
        "twist": (-30.0, 30.0)
    },
    "RightForeArm": {
        "flexion": (0.0, 145.0),
        "twist": (-30.0, 30.0)
    },
    # Shoulder: Natural human range of motion
    "LeftArm": {
        "flexion": (-45.0, 170.0),
        "abduction": (0.0, 170.0),
        "rotation": (-80.0, 80.0)
    },
    "RightArm": {
        "flexion": (-45.0, 170.0),
        "abduction": (0.0, 170.0),
        "rotation": (-80.0, 80.0)
    },
    # Wrist: Natural flexion/extension and radial/ulnar deviation
    "LeftHand": {
        "flexion": (-70.0, 80.0),
        "deviation": (-30.0, 45.0)
    },
    "RightHand": {
        "flexion": (-70.0, 80.0),
        "deviation": (-30.0, 45.0)
    },
    # Finger Knuckles & Phalanges (MCP, PIP, DIP)
    # Fingers are pure hinge flexion joints with zero backward hyperextension
    "Finger_MCP": {
        "flexion": (-10.0, 90.0)
    },
    "Finger_PIP": {
        "flexion": (0.0, 110.0) # Cannot bend backward or twist
    },
    "Finger_DIP": {
        "flexion": (0.0, 90.0)  # Cannot bend backward or twist
    },
    "Thumb_MCP": {
        "flexion": (0.0, 70.0)
    },
    "Thumb_IP": {
        "flexion": (0.0, 90.0)
    }
}

@dataclass
class QualityThresholds:
    """Explicit numeric thresholds for automated quality classification."""
    MIN_HAND_COVERAGE_CLEAN: float = 80.0
    MIN_HAND_COVERAGE_REVIEW: float = 50.0
    MIN_HAND_COVERAGE_FAILED: float = 20.0

    MIN_FACE_COVERAGE_CLEAN: float = 90.0
    MIN_FACE_COVERAGE_REVIEW: float = 70.0
    MIN_FACE_COVERAGE_FAILED: float = 40.0

    MIN_POSE_COVERAGE_CLEAN: float = 90.0
    MIN_POSE_COVERAGE_REVIEW: float = 70.0

    MIN_MOTION_STD_CLEAN: float = 0.5
    MIN_MOTION_STD_FROZEN: float = 0.05

    MIN_FRAMES: int = 5
    
    # Validation gate thresholds
    MAX_CLAMPED_FRAMES_PCT_CLEAN: float = 10.0 # <10% clamping is clean
    MAX_CLAMPED_FRAMES_PCT_REVIEW: float = 30.0 # 10-30% requires review
