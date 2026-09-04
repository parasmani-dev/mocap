"""Configuration parameters, skeleton definitions, and QA thresholds for ISL Mocap Pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Any
from pipeline.bvh_generator import BVH_MIXAMO_HIERARCHY

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

BVH_HIERARCHY = BVH_MIXAMO_HIERARCHY
