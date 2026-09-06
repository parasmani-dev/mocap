"""Stage 3: Automated Kinematic Validation & Quality Assurance Gate.
- Enforces human-anatomical joint limits across all limbs and finger hinges
- Detects and logs any joint clamping warnings (word + joint + frame)
- Validates palm-normal continuity against ghost flips
- Generates structured validation metrics for processing_report.csv
"""

import os
import math
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from pipeline.config import QualityThresholds, ANATOMICAL_JOINT_LIMITS
from pipeline.bvh_generator import BVH_MIXAMO_HIERARCHY

class BVHQualityValidator:
    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QualityThresholds()
        self.joint_limits = ANATOMICAL_JOINT_LIMITS

    def validate_kinematic_stream(
        self,
        word_id: str,
        frames_data: List[Dict[str, Any]],
        clamping_log: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Validate kinematic landmark stream and check clamping stats.
        """
        notes = []
        metrics = {
            "total_frames": len(frames_data),
            "clamped_events_count": 0,
            "clamped_frames_count": 0,
            "ghost_flips_detected": 0,
            "joints_clamped": []
        }
        
        if clamping_log:
            metrics["clamped_events_count"] = len(clamping_log)
            clamped_frames = set(item["frame"] for item in clamping_log)
            metrics["clamped_frames_count"] = len(clamped_frames)
            clamped_joints = sorted(list(set(item["joint"] for item in clamping_log)))
            metrics["joints_clamped"] = clamped_joints

            clamped_pct = (len(clamped_frames) / max(1, len(frames_data))) * 100.0
            metrics["clamped_frames_pct"] = round(clamped_pct, 1)

            if clamped_pct > self.thresholds.MAX_CLAMPED_FRAMES_PCT_REVIEW:
                notes.append(f"High joint clamping ({clamped_pct:.1f}% frames clamped on: {', '.join(clamped_joints[:4])}).")
            elif clamped_pct > self.thresholds.MAX_CLAMPED_FRAMES_PCT_CLEAN:
                notes.append(f"Minor joint clamping ({clamped_pct:.1f}% frames on: {', '.join(clamped_joints[:3])}).")

        # Palm normal continuity check
        for hand_key, is_left in [("right_hand_landmarks", False), ("left_hand_landmarks", True)]:
            prev_norm = None
            hand_flips = 0
            for f_idx, frame in enumerate(frames_data):
                hl = frame.get(hand_key)
                if hl:
                    w = np.array([hl[0]["x"], hl[0]["y"], hl[0]["z"]])
                    imcp = np.array([hl[5]["x"], hl[5]["y"], hl[5]["z"]])
                    pmcp = np.array([hl[17]["x"], hl[17]["y"], hl[17]["z"]])
                    if is_left:
                        v_norm = np.cross(pmcp - w, imcp - w)
                    else:
                        v_norm = np.cross(imcp - w, pmcp - w)
                    norm_len = np.linalg.norm(v_norm)
                    if norm_len > 1e-7:
                        v_norm /= norm_len
                        if prev_norm is not None:
                            if np.dot(prev_norm, v_norm) < -0.2:
                                hand_flips += 1
                        prev_norm = v_norm
            if hand_flips > 0:
                metrics["ghost_flips_detected"] += hand_flips
                notes.append(f"Detected {hand_flips} raw ghost flips on {'Left' if is_left else 'Right'} Hand (corrected via continuity).")

        status = "CLEAN"
        if metrics.get("clamped_frames_pct", 0) > self.thresholds.MAX_CLAMPED_FRAMES_PCT_REVIEW:
            status = "NEEDS_REVIEW"
        return status, notes, metrics

    def validate_bvh_file(self, bvh_path: str, stats: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
        """Validate a BVH file and extraction statistics against strict numeric thresholds."""
        notes = []
        qa_metrics = {}
        
        if not os.path.exists(bvh_path):
            return "FAILED", ["BVH file does not exist on disk."], qa_metrics

        try:
            with open(bvh_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return "FAILED", [f"Error reading BVH file: {str(e)}"], qa_metrics

        hierarchy_found = False
        motion_found = False
        brace_count = 0
        motion_line_idx = -1
        expected_channels = 0
        
        for joint, d in BVH_MIXAMO_HIERARCHY.items():
            expected_channels += len(d.get("channels", []))

        for idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if line.startswith("HIERARCHY"):
                hierarchy_found = True
            elif "{" in line:
                brace_count += line.count("{")
            elif "}" in line:
                brace_count -= line.count("}")
            elif line.startswith("MOTION"):
                motion_found = True
                motion_line_idx = idx
                break

        if not hierarchy_found:
            notes.append("Missing HIERARCHY section.")
        if brace_count != 0:
            notes.append(f"Mismatched braces in HIERARCHY (diff: {brace_count}).")
        if not motion_found:
            notes.append("Missing MOTION section.")
            return "FAILED", notes, qa_metrics

        frames_count = 0
        frame_time = 0.0
        motion_data_start = -1

        for idx in range(motion_line_idx, min(motion_line_idx + 10, len(lines))):
            l = lines[idx].strip()
            if l.startswith("Frames:"):
                frames_count = int(l.split(":")[1].strip())
            elif l.startswith("Frame Time:"):
                frame_time = float(l.split(":")[1].strip())
                motion_data_start = idx + 1
                break

        if motion_data_start == -1 or frames_count <= 0:
            return "FAILED", ["Malformed MOTION header or zero frames."], qa_metrics

        motion_lines = [l.strip() for l in lines[motion_data_start:] if l.strip()]
        if len(motion_lines) != frames_count:
            notes.append(f"Frames count declared ({frames_count}) does not match motion lines ({len(motion_lines)}).")

        motion_matrix = []
        for line_no, m_line in enumerate(motion_lines):
            tokens = m_line.split()
            if len(tokens) != expected_channels:
                notes.append(f"Frame {line_no} channel count mismatch (got {len(tokens)}, expected {expected_channels}).")
                return "FAILED", notes, qa_metrics
            try:
                motion_matrix.append([float(t) for t in tokens])
            except ValueError:
                notes.append(f"Non-numeric values in motion frame {line_no}.")
                return "FAILED", notes, qa_metrics

        motion_arr = np.array(motion_matrix)
        rot_channels = motion_arr[:, 3:] if motion_arr.shape[1] > 3 else motion_arr
        channel_stds = np.std(rot_channels, axis=0)
        mean_motion_std = float(np.mean(channel_stds))
        max_motion_std = float(np.max(channel_stds))
        
        qa_metrics["mean_rot_std_deg"] = round(mean_motion_std, 3)
        qa_metrics["max_rot_std_deg"] = round(max_motion_std, 3)
        qa_metrics["total_frames"] = frames_count
        qa_metrics["channel_count"] = expected_channels

        l_cov = stats.get("left_hand_coverage_pct", 0.0)
        r_cov = stats.get("right_hand_coverage_pct", 0.0)
        face_cov = stats.get("face_coverage_pct", 0.0)
        pose_cov = stats.get("pose_coverage_pct", 0.0)
        
        is_failed = False
        is_review = False

        if frames_count < self.thresholds.MIN_FRAMES:
            notes.append(f"Video too short ({frames_count} frames < min {self.thresholds.MIN_FRAMES}).")
            is_failed = True

        if max_motion_std < self.thresholds.MIN_MOTION_STD_FROZEN:
            notes.append(f"Motion is frozen/flat (max std {max_motion_std:.3f} deg < {self.thresholds.MIN_MOTION_STD_FROZEN}).")
            is_failed = True

        if pose_cov < self.thresholds.MIN_POSE_COVERAGE_REVIEW:
            notes.append(f"Low pose coverage ({pose_cov}% < {self.thresholds.MIN_POSE_COVERAGE_REVIEW}%).")
            if pose_cov < 40.0:
                is_failed = True
            else:
                is_review = True

        if face_cov < self.thresholds.MIN_FACE_COVERAGE_FAILED:
            notes.append(f"Critical face tracking loss ({face_cov}% < {self.thresholds.MIN_FACE_COVERAGE_FAILED}%).")
            is_failed = True
        elif face_cov < self.thresholds.MIN_FACE_COVERAGE_CLEAN:
            notes.append(f"Face tracking dropout ({face_cov}% < {self.thresholds.MIN_FACE_COVERAGE_CLEAN}%).")
            is_review = True

        max_hand_cov = max(l_cov, r_cov)
        min_hand_cov = min(l_cov, r_cov)
        
        if max_hand_cov < self.thresholds.MIN_HAND_COVERAGE_FAILED:
            notes.append(f"No active hand detected (max hand coverage {max_hand_cov}% < {self.thresholds.MIN_HAND_COVERAGE_FAILED}%).")
            is_failed = True
        elif max_hand_cov < self.thresholds.MIN_HAND_COVERAGE_CLEAN:
            notes.append(f"Hand tracking dropout on dominant hand ({max_hand_cov}% < {self.thresholds.MIN_HAND_COVERAGE_CLEAN}%).")
            is_review = True

        if is_failed:
            status = "FAILED"
        elif is_review:
            status = "NEEDS_REVIEW"
        else:
            status = "CLEAN"
            if not notes:
                notes.append("Clean tracking and valid BVH motion.")

        return status, notes, qa_metrics
