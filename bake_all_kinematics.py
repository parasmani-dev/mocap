import os
import sys
import glob
import json
import time
import subprocess

BASE_DIR = r"C:\Users\paras\.gemini\antigravity\scratch\isl_mocap_pipeline"
CHAR_FBX_PATH = r"C:\Users\paras\Downloads\Ch22_nonPBR.fbx"
OUTPUT_LANDMARKS_DIR = os.path.join(BASE_DIR, "output", "landmarks_json")
OUTPUT_GLB_DIR = os.path.join(BASE_DIR, "output", "glb")
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
SOLVER_SCRIPT = os.path.join(BASE_DIR, "pipeline", "kinematic_baker.py")

os.makedirs(OUTPUT_GLB_DIR, exist_ok=True)

# Collect unique canonical word files (skip _raw, _authentic, _trimmed duplicates)
json_files = glob.glob(os.path.join(OUTPUT_LANDMARKS_DIR, "*.json"))
word_dict = {}

for jf in json_files:
    base = os.path.splitext(os.path.basename(jf))[0]
    canonical = base
    for sfx in ["_raw", "_authentic", "_trimmed"]:
        if canonical.endswith(sfx):
            canonical = canonical[:-len(sfx)]
    
    if canonical not in word_dict:
        word_dict[canonical] = jf
    else:
        # Prefer non-raw version if available
        if "_raw" not in base and "_raw" in word_dict[canonical]:
            word_dict[canonical] = jf

print(f"Found {len(word_dict)} unique sign words to bake:")
for w in sorted(word_dict.keys()):
    print(f" - {w}")

success_count = 0
failed_words = []

for idx, (word, json_path) in enumerate(sorted(word_dict.items()), 1):
    glb_path = os.path.join(OUTPUT_GLB_DIR, f"avatar_{word}.glb")
    print(f"\n[{idx}/{len(word_dict)}] Baking {word} -> {glb_path}...")
    t0 = time.time()
    
    cmd = [
        BLENDER_EXE, "--background",
        "--python", SOLVER_SCRIPT,
        "--",
        "--character", CHAR_FBX_PATH,
        "--landmarks", json_path,
        "--output", glb_path,
        "--anim_name", f"sign_{word}"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(glb_path) and os.path.getsize(glb_path) > 100000:
        sz_mb = round(os.path.getsize(glb_path) / (1024 * 1024), 2)
        print(f"  [OK] {word} baked successfully ({sz_mb} MB) in {round(time.time() - t0, 1)}s")
        success_count += 1
    else:
        print(f"  [FAILED] {word} failed to bake")
        failed_words.append(word)

print(f"\n==========================================")
print(f"BATCH BAKE COMPLETE: {success_count}/{len(word_dict)} succeeded")
if failed_words:
    print(f"Failed: {failed_words}")
print(f"==========================================")
