"""Stage 3: Automated Quality Assurance for BVH Files & Landmark Extraction."""

import os
import numpy as np
from typing import Dict, Any, Tuple, List

from pipeline.config import QualityThresholds, BVH_HIERARCHY

class BVHQualityValidator:
    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QualityThresholds()

    def validate_bvh_file(self, bvh_path: str, stats: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Validate a BVH file and extraction statistics against strict numeric thresholds.
        
        Returns:
            (status, notes_list, qa_metrics)
            where status is one of ['CLEAN', 'NEEDS_REVIEW', 'FAILED']
        """
        notes = []
        qa_metrics = {}
        
        if not os.path.exists(bvh_path):
            return "FAILED", ["BVH file does not exist on disk."], qa_metrics

        try:
            with open(bvh_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return "FAILED", [f"Error reading BVH file: {str(e)}"], qa_metrics

        # 1. Syntax & Hierarchy validation
        hierarchy_found = False
        motion_found = False
        brace_count = 0
        motion_line_idx = -1
        expected_channels = 0
        
        # Calculate expected channels from config
        for joint, d in BVH_HIERARCHY.items():
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

        # Parse motion metadata
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

        # Parse motion matrix for variance check
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

        motion_arr = np.array(motion_matrix) # shape: (num_frames, expected_channels)
        
        # Calculate motion variance across all non-root position channels (std dev in degrees)
        # Root positions are indices 0, 1, 2
        rot_channels = motion_arr[:, 3:] if motion_arr.shape[1] > 3 else motion_arr
        channel_stds = np.std(rot_channels, axis=0)
        mean_motion_std = float(np.mean(channel_stds))
        max_motion_std = float(np.max(channel_stds))
        
        qa_metrics["mean_rot_std_deg"] = round(mean_motion_std, 3)
        qa_metrics["max_rot_std_deg"] = round(max_motion_std, 3)
        qa_metrics["total_frames"] = frames_count
        qa_metrics["channel_count"] = expected_channels

        # 2. Check coverage and motion against explicit thresholds
        l_cov = stats.get("left_hand_coverage_pct", 0.0)
        r_cov = stats.get("right_hand_coverage_pct", 0.0)
        face_cov = stats.get("face_coverage_pct", 0.0)
        pose_cov = stats.get("pose_coverage_pct", 0.0)
        
        # Determine status
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

        # Face tracking check
        if face_cov < self.thresholds.MIN_FACE_COVERAGE_FAILED:
            notes.append(f"Critical face tracking loss ({face_cov}% < {self.thresholds.MIN_FACE_COVERAGE_FAILED}%).")
            is_failed = True
        elif face_cov < self.thresholds.MIN_FACE_COVERAGE_CLEAN:
            notes.append(f"Face tracking dropout ({face_cov}% < {self.thresholds.MIN_FACE_COVERAGE_CLEAN}%).")
            is_review = True

        # Hand tracking check (either one dominant hand or both)
        max_hand_cov = max(l_cov, r_cov)
        min_hand_cov = min(l_cov, r_cov)
        
        if max_hand_cov < self.thresholds.MIN_HAND_COVERAGE_FAILED:
            notes.append(f"No active hand detected (max hand coverage {max_hand_cov}% < {self.thresholds.MIN_HAND_COVERAGE_FAILED}%).")
            is_failed = True
        elif max_hand_cov < self.thresholds.MIN_HAND_COVERAGE_CLEAN:
            notes.append(f"Hand tracking dropout on dominant hand ({max_hand_cov}% < {self.thresholds.MIN_HAND_COVERAGE_CLEAN}%).")
            is_review = True
            
        if min_hand_cov < self.thresholds.MIN_HAND_COVERAGE_REVIEW and max_hand_cov >= self.thresholds.MIN_HAND_COVERAGE_CLEAN:
            # Likely a one-handed sign
            notes.append(f"One-handed sign detected (L: {l_cov}%, R: {r_cov}%).")
        elif min_hand_cov >= self.thresholds.MIN_HAND_COVERAGE_REVIEW and min_hand_cov < self.thresholds.MIN_HAND_COVERAGE_CLEAN:
            notes.append(f"Secondary hand has partial coverage ({min_hand_cov}%).")
            is_review = True

        if max_motion_std < self.thresholds.MIN_MOTION_STD_CLEAN and not is_failed:
            notes.append(f"Subtle motion detected (max std {max_motion_std:.3f} deg < {self.thresholds.MIN_MOTION_STD_CLEAN}).")
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
