"""Generate synthetic video clips for automated test suite verification."""

import cv2
import numpy as np
import os
import math

def create_synthetic_sign_video(output_path: str, duration_sec: float = 2.0, fps: int = 30):
    """Generate a synthetic video simulating an ISL signer waving and gesturing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration_sec * fps)
    
    for i in range(total_frames):
        # Create dark background
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        
        t = i / float(fps)
        
        # Signer Head (skin tone)
        cv2.circle(frame, (320, 140), 45, (180, 210, 240), -1)
        # Eyes & mouth for face mesh
        cv2.circle(frame, (305, 130), 5, (50, 50, 50), -1)
        cv2.circle(frame, (335, 130), 5, (50, 50, 50), -1)
        cv2.ellipse(frame, (320, 160), (15, 8), 0, 0, 180, (50, 50, 180), -1)
        
        # Torso
        cv2.rectangle(frame, (250, 190), (390, 360), (120, 80, 50), -1)
        
        # Right Arm & Hand moving in sign gesture
        r_hand_x = int(320 + 80 * math.sin(2 * math.pi * t))
        r_hand_y = int(220 + 60 * math.cos(2 * math.pi * t))
        
        # Upper arm & forearm
        cv2.line(frame, (260, 210), (220, 270), (120, 80, 50), 16)
        cv2.line(frame, (220, 270), (r_hand_x, r_hand_y), (120, 80, 50), 14)
        
        # Hand & fingers
        cv2.circle(frame, (r_hand_x, r_hand_y), 18, (180, 210, 240), -1)
        for finger_idx in range(5):
            f_angle = finger_idx * 0.3 - 0.6 + 0.2 * math.sin(4 * math.pi * t)
            fx = int(r_hand_x + 25 * math.cos(f_angle))
            fy = int(r_hand_y + 25 * math.sin(f_angle))
            cv2.line(frame, (r_hand_x, r_hand_y), (fx, fy), (180, 210, 240), 5)
            
        # Left Arm & Hand resting or complementary sign
        l_hand_x = int(400 + 40 * math.cos(2 * math.pi * t))
        l_hand_y = int(260 + 30 * math.sin(2 * math.pi * t))
        cv2.line(frame, (380, 210), (420, 270), (120, 80, 50), 16)
        cv2.line(frame, (420, 270), (l_hand_x, l_hand_y), (120, 80, 50), 14)
        cv2.circle(frame, (l_hand_x, l_hand_y), 16, (180, 210, 240), -1)
        for finger_idx in range(5):
            f_angle = finger_idx * 0.3 + 1.8
            fx = int(l_hand_x + 22 * math.cos(f_angle))
            fy = int(l_hand_y + 22 * math.sin(f_angle))
            cv2.line(frame, (l_hand_x, l_hand_y), (fx, fy), (180, 210, 240), 4)

        out.write(frame)
        
    out.release()
    print(f"Synthetic video generated: {output_path} ({total_frames} frames)")
