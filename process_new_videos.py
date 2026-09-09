"""Automated Batch Ingestion Script for New / Remaining ISL Videos.
Processes all videos in dataset/new_videos/ and saves results into output/new_batch/
leaving the verified master library untouched.
"""

import os
import sys
import glob
import json
import subprocess
import time
from pathlib import Path

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
CHARACTER_FBX = r"C:\Users\paras\Downloads\Ch22_nonPBR.fbx"

BASE_DIR = Path(r"C:\Users\paras\.gemini\antigravity\scratch\isl_mocap_pipeline")
IN_DIR = BASE_DIR / "dataset" / "new_videos"
OUT_BASE = BASE_DIR / "output" / "new_batch"
OUT_JSON = OUT_BASE / "landmarks_json"
OUT_GLB = OUT_BASE / "glb"
OUT_CLAMPS = OUT_BASE / "clamping_logs"
MANIFEST_FILE = OUT_BASE / "manifest.json"

for d in [IN_DIR, OUT_JSON, OUT_GLB, OUT_CLAMPS]:
    d.mkdir(parents=True, exist_ok=True)

def process_single_video(video_path: Path):
    word_id = video_path.stem.lower().replace(" ", "_").replace("-", "_")
    print(f"\n==================================================")
    print(f"[*] Processing New Video: {video_path.name} -> ID: {word_id}")
    print(f"==================================================")
    
    json_path = OUT_JSON / f"{word_id}.json"
    glb_path = OUT_GLB / f"avatar_{word_id}.glb"
    clamp_log = OUT_CLAMPS / f"{word_id}_clamps.json"
    
    # 1. MediaPipe Extraction
    if not json_path.exists():
        print(f"  [1/2] Running MediaPipe Holistic Extraction...")
        extract_cmd = [
            sys.executable,
            str(BASE_DIR / "pipeline" / "extractor.py"),
            "--input", str(video_path),
            "--output", str(json_path)
        ]
        res = subprocess.run(extract_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] MediaPipe extraction failed: {res.stderr}")
            return False
    else:
        print(f"  [1/2] Using existing landmarks JSON: {json_path.name}")
        
    # 2. Blender Kinematic Solver & NLA glTF Baker
    print(f"  [2/2] Running Kinematic Baker & Relative Rest Solver...")
    bake_cmd = [
        BLENDER_EXE,
        "-b",
        "--python", str(BASE_DIR / "pipeline" / "kinematic_baker.py"),
        "--",
        "--character", CHARACTER_FBX,
        "--landmarks", str(json_path),
        "--output", str(glb_path),
        "--anim_name", word_id,
        "--clamping_log", str(clamp_log)
    ]
    res = subprocess.run(bake_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [ERROR] Blender baking failed: {res.stderr}")
        return False
        
    size_mb = glb_path.stat().st_size / (1024 * 1024) if glb_path.exists() else 0.0
    print(f"  [SUCCESS] Baked GLB: {glb_path.name} ({size_mb:.2f} MB)")
    return True

def update_manifest():
    glb_files = list(OUT_GLB.glob("avatar_*.glb"))
    signs = []
    for g in sorted(glb_files):
        word_id = g.stem.replace("avatar_", "")
        display = word_id.replace("_", " ").title()
        signs.append({
            "id": word_id,
            "label": display,
            "glb_path": f"output/new_batch/glb/{g.name}"
        })
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({"batch": "new_incoming", "count": len(signs), "signs": signs}, f, indent=2)
    print(f"\n[+] Updated new batch manifest: {MANIFEST_FILE} ({len(signs)} signs)")

def main():
    video_exts = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MOV", "*.MP4"]
    videos = []
    for ext in video_exts:
        videos.extend(list(IN_DIR.glob(ext)))
        
    if not videos:
        print(f"[!] No new videos found in {IN_DIR}")
        print(f"    Please place your video files (.mp4 / .mov) into:")
        print(f"    {IN_DIR}")
        return
        
    print(f"Found {len(videos)} new video(s) to process in {IN_DIR}")
    for idx, v in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}]")
        process_single_video(v)
        
    update_manifest()
    print("\n[✔] Processing completed for new batch.")

if __name__ == "__main__":
    main()
