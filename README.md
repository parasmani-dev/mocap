# End-to-End ISL Motion Capture & 3D Avatar Generation Pipeline

An automated AI pipeline converting monocular Indian Sign Language (ISL) video footage into kinematically compliant BioVision Hierarchy (.bvh) skeletal motion files and web-optimized 3D avatar animations (.glb) with zero lag.

---

## 🚀 Key Highlights & Architecture

1. **Holistic Landmark Extraction**: MediaPipe Holistic (Pose + 42 Hand Keypoints + Face mesh) with temporal Outlier & Missing Frame interpolation.
2. **Anatomical Kinematics Engine**:
   - Upright posture locking (vertical hip/spine stabilization with zero slouch).
   - Natural straight standing lower body.
   - Head gaze locked forward and upright.
   - 1-DOF biological finger flexion with orthogonal palm reference frames.
3. **Blender Matrix Delta Retargeter**:
   - Mathematical retargeting mapping arbitrary BVH motion to 65-bone Mixamo standard armatures.
   - Fully automated NLA (Non-Linear Animation) strip baking.
   - Texture compression and downscaling (63 MB -> 4.2 MB web-ready GLBs).
4. **Three.js Web Avatar Studio**:
   - Real-time 60 FPS 3D sign language playback interface.
   - Complete dictionary manifest and interactive animation switcher.

---

## 📁 Repository Structure

`
├── pipeline/
│   ├── extractor.py         # MediaPipe Holistic keypoint extraction
│   ├── smoothing.py         # Euro-filter temporal smoothing & outlier rejection
│   ├── bvh_generator.py     # Kinematics engine & BioVision BVH exporter
│   ├── retarget_robust.py   # Matrix delta armature retargeter
│   ├── bake_nla_gltf.py     # Blender NLA baking and glTF exporter
│   └── validator.py         # Kinematic and boundary validator
├── output/
│   ├── bvh/                 # Generated BVH motion files (31 signs)
│   ├── glb/                 # Web-ready animated character models (31 signs)
│   └── signs_manifest.json  # Web catalog metadata
├── index.html               # Three.js 3D Avatar Web Studio
├── run_pipeline.py          # Single-video CLI runner
└── batch_rebake.py          # Batch automation script
`

---

## 🛠️ Requirements & Setup

- Python 3.10+ (mediapipe, opencv-python, scipy, 
umpy)
- Blender 4.x / 5.x (for background NLA baking)
- Any modern web browser supporting WebGL2
