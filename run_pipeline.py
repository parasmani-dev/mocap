"""ISL Mocap Pipeline CLI - Batch and Single-Word Processing Entrypoint.
Supports full end-to-end video -> MediaPipe landmarks -> Kinematic BVH -> Direct Blender GLB baking.
"""

import os
import sys
import glob
import json
import csv
import argparse
import subprocess
import time
from typing import List, Dict, Any

from pipeline.config import QualityThresholds
from pipeline.extractor import LandmarkExtractor
from pipeline.bvh_generator import BVHConverter
from pipeline.validator import BVHQualityValidator

DEFAULT_BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
DEFAULT_CHAR_FBX = r"C:\Users\paras\Downloads\Ch22_nonPBR.fbx"

def sanitize_word_id(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    for sfx in ["_raw", "_authentic", "_trimmed"]:
        if base.endswith(sfx):
            base = base[:-len(sfx)]
    return base.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("'", "")

def process_single_video(
    video_path: str,
    output_dir: str,
    extractor: LandmarkExtractor,
    bvh_converter: BVHConverter,
    validator: BVHQualityValidator,
    bake_glb: bool = False,
    blender_exe: str = DEFAULT_BLENDER_EXE,
    char_fbx: str = DEFAULT_CHAR_FBX,
    force: bool = False
) -> Dict[str, Any]:
    """Process a single video file through all stages: extraction, BVH conversion, and optional GLB baking."""
    filename = os.path.basename(video_path)
    word = sanitize_word_id(filename)
    
    landmarks_dir = os.path.join(output_dir, "landmarks_json")
    bvh_dir = os.path.join(output_dir, "bvh")
    glb_dir = os.path.join(output_dir, "glb")
    
    json_path = os.path.join(landmarks_dir, f"{word}.json")
    bvh_path = os.path.join(bvh_dir, f"{word}.bvh")
    glb_path = os.path.join(glb_dir, f"avatar_{word}.glb")
    
    os.makedirs(landmarks_dir, exist_ok=True)
    os.makedirs(bvh_dir, exist_ok=True)
    if bake_glb:
        os.makedirs(glb_dir, exist_ok=True)
    
    print(f"\n[{word}] Starting processing: {video_path}")
    t0 = time.time()
    
    # Stage 1: Landmark Extraction
    try:
        if os.path.exists(json_path) and not force:
            print(f"[{word}] Loading cached landmarks from {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                landmark_data = json.load(f)
            stats = landmark_data.get("stats", {})
        else:
            landmark_data, stats = extractor.process_video(video_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(landmark_data, f, indent=2)
            print(f"[{word}] Stage 1 complete ({stats['processing_time_sec']}s, {stats['total_frames']} frames, {stats['fps_throughput']} fps)")
            print(f"[{word}] Coverage: Pose={stats['pose_coverage_pct']}%, Face={stats['face_coverage_pct']}%, LHand={stats['left_hand_coverage_pct']}%, RHand={stats['right_hand_coverage_pct']}%")
    except Exception as e:
        print(f"[{word}] Stage 1 FAILED: {str(e)}")
        return {
            "word_id": word,
            "status": "FAILED",
            "total_frames": 0,
            "fps": 0,
            "pose_coverage_pct": 0.0,
            "face_coverage_pct": 0.0,
            "left_hand_coverage_pct": 0.0,
            "right_hand_coverage_pct": 0.0,
            "ghost_flips_corrected": 0,
            "clamped_frames_pct": 0.0,
            "clamped_events_count": 0,
            "joints_clamped": "",
            "glb_size_mb": 0.0,
            "processing_time_sec": 0.0,
            "notes": f"Stage 1 Extraction Error: {str(e)}"
        }

    # Stage 2: Kinematic BVH Generation
    try:
        bvh_converter.convert_landmarks_to_bvh(landmark_data, bvh_path, smooth=True)
        print(f"[{word}] Stage 2 BVH generation complete -> {bvh_path}")
    except Exception as e:
        print(f"[{word}] Stage 2 FAILED: {str(e)}")
        return {
            "word_id": word,
            "status": "FAILED",
            "total_frames": stats.get("total_frames", 0),
            "fps": stats.get("fps", 0),
            "pose_coverage_pct": stats.get("pose_coverage_pct", 0.0),
            "face_coverage_pct": stats.get("face_coverage_pct", 0.0),
            "left_hand_coverage_pct": stats.get("left_hand_coverage_pct", 0.0),
            "right_hand_coverage_pct": stats.get("right_hand_coverage_pct", 0.0),
            "ghost_flips_corrected": 0,
            "clamped_frames_pct": 0.0,
            "clamped_events_count": 0,
            "joints_clamped": "",
            "glb_size_mb": 0.0,
            "processing_time_sec": stats.get("processing_time_sec", 0.0),
            "notes": f"Stage 2 BVH Conversion Error: {str(e)}"
        }

    # Stage 3: Kinematic Validation Gate
    frames = landmark_data.get("frames", [])
    status, notes, metrics = validator.validate_kinematic_stream(word, frames, bvh_converter.clamping_log)
    print(f"[{word}] Stage 3 Validation Gate: {status} | Ghost Flips: {metrics['ghost_flips_detected']} | Clamped Frames: {metrics.get('clamped_frames_pct', 0.0)}%")

    # Stage 4 (Optional): Blender Direct GLB Baking
    glb_size_mb = 0.0
    if bake_glb:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        solver_script = os.path.join(base_dir, "pipeline", "kinematic_baker.py")
        if os.path.exists(blender_exe) and os.path.exists(char_fbx) and os.path.exists(solver_script):
            print(f"[{word}] Stage 4 Baking GLB avatar...")
            cmd = [
                blender_exe, "--background",
                "--python", solver_script,
                "--",
                "--character", char_fbx,
                "--landmarks", json_path,
                "--output", glb_path,
                "--anim_name", f"sign_{word}"
            ]
            subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(glb_path):
                glb_size_mb = round(os.path.getsize(glb_path) / (1024 * 1024), 2)
                print(f"[{word}] Stage 4 complete -> {glb_path} ({glb_size_mb} MB)")
            else:
                notes.append("Blender GLB bake did not output file")
        else:
            notes.append("Blender executable or Character FBX path not found for Stage 4")

    total_time = round(time.time() - t0, 2)
    return {
        "word_id": word,
        "status": status,
        "total_frames": stats.get("total_frames", len(frames)),
        "fps": stats.get("fps", 30.0),
        "pose_coverage_pct": stats.get("pose_coverage_pct", 100.0),
        "face_coverage_pct": stats.get("face_coverage_pct", 100.0),
        "left_hand_coverage_pct": stats.get("left_hand_coverage_pct", 0.0),
        "right_hand_coverage_pct": stats.get("right_hand_coverage_pct", 0.0),
        "ghost_flips_corrected": metrics.get("ghost_flips_detected", 0),
        "clamped_frames_pct": metrics.get("clamped_frames_pct", 0.0),
        "clamped_events_count": metrics.get("clamped_events_count", 0),
        "joints_clamped": "; ".join(metrics.get("joints_clamped", [])),
        "glb_size_mb": glb_size_mb,
        "processing_time_sec": total_time,
        "notes": " | ".join(notes) if notes else "Clean anatomical motion"
    }

def update_manifest(output_dir: str):
    """Scan output/glb and output/bvh to build/update signs_manifest.json."""
    glb_dir = os.path.join(output_dir, "glb")
    bvh_dir = os.path.join(output_dir, "bvh")
    manifest_path = os.path.join(output_dir, "signs_manifest.json")
    
    if not os.path.exists(glb_dir):
        return
        
    glb_files = sorted(glob.glob(os.path.join(glb_dir, "avatar_*.glb")))
    manifest = []
    
    for g_path in glb_files:
        base = os.path.basename(g_path)
        word_id = base.replace("avatar_", "").replace(".glb", "")
        bvh_file = f"./output/bvh/{word_id}.bvh"
        
        manifest.append({
            "word_id": word_id,
            "label": word_id.replace("_", " ").title(),
            "glb_file": f"./output/glb/{base}",
            "bvh_file": bvh_file if os.path.exists(os.path.join(output_dir, "bvh", f"{word_id}.bvh")) else "",
            "duration_sec": 2.0,
            "status": "CLEAN"
        })
        
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Signs manifest updated with {len(manifest)} signs -> {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="ISL Video to Kinematic BVH & 3D Avatar GLB Pipeline")
    parser.add_argument("--input_dir", type=str, default="./input_videos", help="Directory containing .mp4 or .mov videos")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--file", type=str, default=None, help="Process single video file")
    parser.add_argument("--word", type=str, default=None, help="Process single word from input_dir")
    parser.add_argument("--model_complexity", type=int, default=2, help="MediaPipe model complexity (0, 1, 2)")
    parser.add_argument("--bake_glb", action="store_true", help="Bake into 3D avatar GLB models via Blender")
    parser.add_argument("--blender_exe", type=str, default=DEFAULT_BLENDER_EXE, help="Path to blender.exe")
    parser.add_argument("--char_fbx", type=str, default=DEFAULT_CHAR_FBX, help="Path to character FBX")
    parser.add_argument("--force", action="store_true", help="Force re-extraction and overwrite")
    
    args = parser.parse_args()
    
    extractor = LandmarkExtractor(model_complexity=args.model_complexity)
    bvh_converter = BVHConverter()
    validator = BVHQualityValidator()
    
    # Collect files
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File does not exist: {args.file}")
            sys.exit(1)
        video_files = [args.file]
    elif args.word:
        candidates = glob.glob(os.path.join(args.input_dir, f"{args.word}.*"))
        if not candidates:
            print(f"Error: Video for word '{args.word}' not found in {args.input_dir}")
            sys.exit(1)
        video_files = [candidates[0]]
    else:
        video_files = []
        for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
            video_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        video_files = sorted(video_files)
        if not video_files:
            print(f"No video files found in {args.input_dir}")
            sys.exit(1)

    print(f"=== ISL Mocap Pipeline: Processing {len(video_files)} video(s) ===")
    os.makedirs(args.output_dir, exist_ok=True)
    
    reports = []
    for v_path in video_files:
        rep = process_single_video(
            video_path=v_path,
            output_dir=args.output_dir,
            extractor=extractor,
            bvh_converter=bvh_converter,
            validator=validator,
            bake_glb=args.bake_glb,
            blender_exe=args.blender_exe,
            char_fbx=args.char_fbx,
            force=args.force
        )
        reports.append(rep)
        
    # Write processing_report.csv
    report_csv = os.path.join(args.output_dir, "processing_report.csv")
    fieldnames = [
        "word_id", "status", "ghost_flips_corrected", "clamped_frames_pct",
        "clamped_events_count", "joints_clamped", "glb_size_mb", "processing_time_sec", "notes"
    ]
    
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in reports:
            writer.writerow(r)
            
    if args.bake_glb:
        update_manifest(args.output_dir)
            
    print(f"\nSummary report saved to: {report_csv}")
    print(f"Total: {len(reports)} | CLEAN: {sum(1 for r in reports if r['status']=='CLEAN')} | NEEDS_REVIEW: {sum(1 for r in reports if r['status']=='NEEDS_REVIEW')} | FAILED: {sum(1 for r in reports if r['status']=='FAILED')}")

if __name__ == "__main__":
    main()
