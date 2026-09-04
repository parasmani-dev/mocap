"""Stage 1: MediaPipe Tasks Landmark Extractor for ISL Videos (Pose Heavy, Hands, Face Mesh)."""

import os
import time
import urllib.request
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pipeline.smoothing import interpolate_missing_landmarks

class LandmarkExtractor:
    def __init__(
        self,
        model_complexity: int = 2,
        models_dir: Optional[str] = None
    ):
        self.model_complexity = model_complexity
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.pose_model_path = os.path.join(self.models_dir, "pose_landmarker.task")
        self.hand_model_path = os.path.join(self.models_dir, "hand_landmarker.task")
        self.face_model_path = os.path.join(self.models_dir, "face_landmarker.task")
        
        self._ensure_models_downloaded()

    def _ensure_models_downloaded(self):
        """Ensure all required MediaPipe task models are downloaded locally."""
        urls = {
            self.pose_model_path: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            self.hand_model_path: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            self.face_model_path: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        }
        for dest, url in urls.items():
            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                print(f"Downloading model {os.path.basename(dest)}...")
                urllib.request.urlretrieve(url, dest)
                print(f"Downloaded {os.path.basename(dest)}.")

    def process_video(self, video_path: str, fill_missing: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process video frame-by-frame and extract complete holistic landmark data.
        Returns (landmark_data, stats)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize Landmarkers with IMAGE mode for highest per-frame accuracy (no video-mode lag/dropouts)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.face_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True
        )
        
        frames_data = []
        pose_present_count = 0
        left_hand_present_count = 0
        right_hand_present_count = 0
        face_present_count = 0
        
        start_time = time.perf_counter()
        
        with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
             vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
             vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                frame_record = {
                    "frame_index": frame_idx,
                    "timestamp": round(frame_idx / fps, 4),
                    "pose_landmarks": None,
                    "pose_world_landmarks": None,
                    "left_hand_landmarks": None,
                    "right_hand_landmarks": None,
                    "face_landmarks": None
                }
                
                # 1. Pose
                pose_res = pose_landmarker.detect(mp_image)
                if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                    pose_present_count += 1
                    frame_record["pose_landmarks"] = [
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, "visibility", 1.0)}
                        for lm in pose_res.pose_landmarks[0]
                    ]
                if pose_res.pose_world_landmarks and len(pose_res.pose_world_landmarks) > 0:
                    frame_record["pose_world_landmarks"] = [
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, "visibility", 1.0)}
                        for lm in pose_res.pose_world_landmarks[0]
                    ]
                    
                # 2. Hands (Left vs Right handedness)
                hand_res = hand_landmarker.detect(mp_image)
                if hand_res.hand_landmarks:
                    for handedness, landmarks in zip(hand_res.handedness, hand_res.hand_landmarks):
                        label = handedness[0].category_name # "Left" or "Right"
                        # Note: MediaPipe detects handedness from mirror perspective
                        lms_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
                        if label == "Left":
                            left_hand_present_count += 1
                            frame_record["left_hand_landmarks"] = lms_list
                        elif label == "Right":
                            right_hand_present_count += 1
                            frame_record["right_hand_landmarks"] = lms_list
                            
                # 3. Face (468/478 Face Mesh)
                face_res = face_landmarker.detect(mp_image)
                if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
                    face_present_count += 1
                    frame_record["face_landmarks"] = [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in face_res.face_landmarks[0]
                    ]
                    
                frames_data.append(frame_record)
                frame_idx += 1

        cap.release()
        elapsed_sec = time.perf_counter() - start_time
        processed_frames = len(frames_data)
        
        pose_coverage = (pose_present_count / processed_frames * 100.0) if processed_frames > 0 else 0.0
        left_hand_coverage = (left_hand_present_count / processed_frames * 100.0) if processed_frames > 0 else 0.0
        right_hand_coverage = (right_hand_present_count / processed_frames * 100.0) if processed_frames > 0 else 0.0
        face_coverage = (face_present_count / processed_frames * 100.0) if processed_frames > 0 else 0.0
        
        video_metadata = {
            "video_path": video_path,
            "filename": os.path.basename(video_path),
            "fps": fps,
            "total_frames": processed_frames,
            "resolution": [width, height],
            "model_complexity": self.model_complexity,
            "processing_time_sec": round(elapsed_sec, 3),
            "fps_throughput": round(processed_frames / elapsed_sec, 2) if elapsed_sec > 0 else 0.0
        }
        
        stats = {
            **video_metadata,
            "pose_coverage_pct": round(pose_coverage, 2),
            "left_hand_coverage_pct": round(left_hand_coverage, 2),
            "right_hand_coverage_pct": round(right_hand_coverage, 2),
            "face_coverage_pct": round(face_coverage, 2),
            "pose_present_frames": pose_present_count,
            "left_hand_present_frames": left_hand_present_count,
            "right_hand_present_frames": right_hand_present_count,
            "face_present_frames": face_present_count
        }
        
        if fill_missing and processed_frames > 0:
            interpolate_missing_landmarks(frames_data, "pose_world_landmarks")
            interpolate_missing_landmarks(frames_data, "pose_landmarks")
            interpolate_missing_landmarks(frames_data, "left_hand_landmarks")
            interpolate_missing_landmarks(frames_data, "right_hand_landmarks")
            interpolate_missing_landmarks(frames_data, "face_landmarks")

        result = {
            "metadata": video_metadata,
            "stats": stats,
            "frames": frames_data
        }
        
        return result, stats
