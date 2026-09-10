"""
Re-bake all GLBs from cached landmark JSONs using the updated kinematic_baker.py.
This applies the straight-finger fix without needing source videos.
"""

import os
import sys
import json
import time
import subprocess

BASE_DIR = r"C:\Users\paras\.gemini\antigravity\scratch\isl_mocap_pipeline"
CHAR_FBX_PATH = r"C:\Users\paras\Downloads\Ch22_nonPBR.fbx"
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
BAKER_SCRIPT = os.path.join(BASE_DIR, "pipeline", "kinematic_baker.py")
LANDMARKS_DIR = os.path.join(BASE_DIR, "output", "landmarks_json")
GLB_DIR = os.path.join(BASE_DIR, "output", "glb")
MANIFEST_PATH = os.path.join(BASE_DIR, "output", "signs_manifest.json")

os.makedirs(GLB_DIR, exist_ok=True)

def get_sign_jsons():
    files = sorted([
        f for f in os.listdir(LANDMARKS_DIR)
        if f.endswith(".json") and "_raw" not in f
    ])
    return files

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return {e["word_id"]: e for e in json.load(f)}
    return {}

def save_manifest(manifest_dict):
    entries = sorted(manifest_dict.values(), key=lambda e: e["word_id"])
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

def main():
    json_files = get_sign_jsons()
    total = len(json_files)
    manifest = load_manifest()

    print("=" * 70)
    print(f"RE-BAKING {total} SIGNS FROM CACHED LANDMARKS (STRAIGHT-FINGER FIX)")
    print("=" * 70)

    ok = 0
    fail = 0

    for idx, fname in enumerate(json_files, 1):
        word_id = os.path.splitext(fname)[0]
        json_path = os.path.join(LANDMARKS_DIR, fname)
        glb_path = os.path.join(GLB_DIR, f"avatar_{word_id}.glb")

        print(f"\n[{idx}/{total}] Baking '{word_id}'...")
        t0 = time.time()

        cmd = [
            BLENDER_EXE, "-b", "-P", BAKER_SCRIPT,
            "--",
            "--character", CHAR_FBX_PATH,
            "--landmarks", json_path,
            "--output", glb_path,
            "--anim_name", word_id,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        dt = time.time() - t0
        output_all = result.stdout + result.stderr

        if "SUCCESSFULLY_EXPORTED" in output_all and os.path.exists(glb_path):
            size_mb = os.path.getsize(glb_path) / 1e6
            print(f"  -> OK ({dt:.1f}s) | {size_mb:.2f} MB")
            ok += 1

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fps = data.get("fps") or data.get("metadata", {}).get("fps", 30.0) or 30.0
            n_frames = len(data.get("frames", []))
            duration = round(n_frames / fps, 2)

            manifest[word_id] = {
                "word_id": word_id,
                "label": word_id.replace("_", " ").title(),
                "glb_file": f"./output/glb/avatar_{word_id}.glb",
                "bvh_file": f"./output/bvh/{word_id}.bvh",
                "duration_sec": duration,
                "status": "CLEAN",
            }
        else:
            print(f"  -> FAILED ({dt:.1f}s)")
            lines = (result.stdout + result.stderr).strip().splitlines()
            for l in lines[-20:]:
                print(f"     {l}")
            fail += 1

        save_manifest(manifest)

    print("\n" + "=" * 70)
    print(f"DONE: {ok} OK, {fail} FAILED -- manifest updated.")
    print("=" * 70)

if __name__ == "__main__":
    main()
