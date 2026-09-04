"""ISL Mocap Pipeline CLI - Batch and Single-Word Processing Entrypoint."""

import os
import sys
import glob
import json
import csv
import argparse
from typing import List, Dict, Any

from pipeline.config import QualityThresholds
from pipeline.extractor import LandmarkExtractor
from pipeline.bvh_generator import BVHConverter
from pipeline.validator import BVHQualityValidator

def process_single_video(
    video_path: str,
    output_dir: str,
    extractor: LandmarkExtractor,
    bvh_converter: BVHConverter,
    validator: BVHQualityValidator,
    force: bool = False
) -> Dict[str, Any]:
    """Process a single video file through all stages."""
    filename = os.path.basename(video_path)
    word = os.path.splitext(filename)[0]
    
    landmarks_dir = os.path.join(output_dir, "landmarks_json")
    bvh_dir = os.path.join(output_dir, "bvh")
    
    json_path = os.path.join(landmarks_dir, f"{word}.json")
    bvh_path = os.path.join(bvh_dir, f"{word}.bvh")
    
    os.makedirs(landmarks_dir, exist_ok=True)
    os.makedirs(bvh_dir, exist_ok=True)
    
    print(f"\n[{word}] Starting processing: {video_path}")
    
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
            "word": word,
            "status": "FAILED",
            "total_frames": 0,
            "fps": 0,
            "pose_coverage_pct": 0.0,
            "face_coverage_pct": 0.0,
            "left_hand_coverage_pct": 0.0,
            "right_hand_coverage_pct": 0.0,
            "processing_time_sec": 0.0,
            "notes": f"Stage 1 Extraction Error: {str(e)}"
        }

    # Stage 2: BVH Generation
    try:
        bvh_converter.convert_landmarks_to_bvh(landmark_data, bvh_path)
        print(f"[{word}] Stage 2 complete -> {bvh_path}")
    except Exception as e:
        print(f"[{word}] Stage 2 FAILED: {str(e)}")
        return {
            "word": word,
            "status": "FAILED",
            "total_frames": stats.get("total_frames", 0),
            "fps": stats.get("fps", 0),
            "pose_coverage_pct": stats.get("pose_coverage_pct", 0.0),
            "face_coverage_pct": stats.get("face_coverage_pct", 0.0),
            "left_hand_coverage_pct": stats.get("left_hand_coverage_pct", 0.0),
            "right_hand_coverage_pct": stats.get("right_hand_coverage_pct", 0.0),
            "processing_time_sec": stats.get("processing_time_sec", 0.0),
            "notes": f"Stage 2 BVH Conversion Error: {str(e)}"
        }

    # Stage 3: Quality Check
    status, notes, qa_metrics = validator.validate_bvh_file(bvh_path, stats)
    print(f"[{word}] Stage 3 QA Status: {status} | Notes: {'; '.join(notes)}")
    
    return {
        "word": word,
        "status": status,
        "total_frames": stats.get("total_frames", 0),
        "fps": stats.get("fps", 0),
        "pose_coverage_pct": stats.get("pose_coverage_pct", 0.0),
        "face_coverage_pct": stats.get("face_coverage_pct", 0.0),
        "left_hand_coverage_pct": stats.get("left_hand_coverage_pct", 0.0),
        "right_hand_coverage_pct": stats.get("right_hand_coverage_pct", 0.0),
        "mean_rot_std_deg": qa_metrics.get("mean_rot_std_deg", 0.0),
        "processing_time_sec": stats.get("processing_time_sec", 0.0),
        "notes": "; ".join(notes)
    }

def main():
    parser = argparse.ArgumentParser(description="ISL Video to BVH Motion Batch Pipeline")
    parser.add_argument("--input_dir", type=str, default="./input_videos", help="Directory containing .mp4 videos")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--file", type=str, default=None, help="Process single video file")
    parser.add_argument("--word", type=str, default=None, help="Process single word from input_dir")
    parser.add_argument("--model_complexity", type=int, default=2, help="MediaPipe model complexity (0, 1, 2)")
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
        target = os.path.join(args.input_dir, f"{args.word}.mp4")
        if not os.path.exists(target):
            print(f"Error: Video for word '{args.word}' not found at {target}")
            sys.exit(1)
        video_files = [target]
    else:
        video_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
        if not video_files:
            print(f"No .mp4 files found in {args.input_dir}")
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
            force=args.force
        )
        reports.append(rep)
        
    # Write processing_report.csv
    report_csv = os.path.join(args.output_dir, "processing_report.csv")
    fieldnames = [
        "word", "status", "total_frames", "fps", "pose_coverage_pct", "face_coverage_pct",
        "left_hand_coverage_pct", "right_hand_coverage_pct", "mean_rot_std_deg",
        "processing_time_sec", "notes"
    ]
    
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            writer.writerow(r)
            
    print(f"\nSummary report saved to: {report_csv}")
    print(f"Total: {len(reports)} | CLEAN: {sum(1 for r in reports if r['status']=='CLEAN')} | NEEDS_REVIEW: {sum(1 for r in reports if r['status']=='NEEDS_REVIEW')} | FAILED: {sum(1 for r in reports if r['status']=='FAILED')}")

if __name__ == "__main__":
    main()
