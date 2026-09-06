"""End-to-end tests for the ISL Mocap Pipeline."""

import os
import unittest
import numpy as np

from pipeline.config import QualityThresholds, BVH_HIERARCHY
from pipeline.bvh_generator import BVHConverter, rotation_matrix_from_vectors, euler_degrees_from_matrix
from pipeline.validator import BVHQualityValidator

class TestMocapPipeline(unittest.TestCase):
    def test_vector_rotation_math(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        mat = rotation_matrix_from_vectors(v1, v2)
        transformed = mat @ v1
        np.testing.assert_allclose(transformed, v2, atol=1e-5)
        
        euler = euler_degrees_from_matrix(mat, "ZXY")
        self.assertAlmostEqual(euler[0], 90.0, places=2)

    def test_bvh_header_generation(self):
        converter = BVHConverter()
        header, joint_order = converter.generate_bvh_header()
        self.assertIn("HIERARCHY", header)
        self.assertIn("ROOT Hips", header)
        self.assertIn("JOINT Spine2", header)
        self.assertIn("JOINT LeftHandThumb1", header)
        self.assertIn("JOINT RightHandPinky3", header)
        
        # Check brace balancing
        self.assertEqual(header.count("{"), header.count("}"))
        self.assertEqual(joint_order[0], "Hips")

    def test_validator_clean_status(self):
        validator = BVHQualityValidator()
        dummy_stats = {
            "pose_coverage_pct": 98.0,
            "face_coverage_pct": 95.0,
            "left_hand_coverage_pct": 88.0,
            "right_hand_coverage_pct": 92.0,
            "total_frames": 30
        }
        
        test_bvh = "test_mock.bvh"
        converter = BVHConverter()
        header, joint_order = converter.generate_bvh_header()
        
        expected_channels = sum(len(d.get("channels", [])) for d in BVH_HIERARCHY.values())
        
        motion_lines = ["MOTION", "Frames: 30", "Frame Time: 0.033333"]
        for i in range(30):
            row = [0.0] * expected_channels
            row[3] = float(i * 2.0)
            row[4] = float(i * 1.5)
            motion_lines.append(" ".join(f"{v:.4f}" for v in row))
            
        with open(test_bvh, "w", encoding="utf-8") as f:
            f.write(header + "\n" + "\n".join(motion_lines) + "\n")
            
        status, notes, metrics = validator.validate_bvh_file(test_bvh, dummy_stats)
        if os.path.exists(test_bvh):
            os.remove(test_bvh)
            
        self.assertEqual(status, "CLEAN")

if __name__ == "__main__":
    unittest.main()
