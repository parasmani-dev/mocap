import os
import glob
import subprocess

# Directories
input_dir = r"G:\Shared drives\Parasmani\college\hackathon\Sih\Avatar\upload here"
output_dir = r"C:\Users\paras\.gemini\antigravity\scratch\isl_mocap_pipeline\output"

# We check what is already in output/glb
existing_glbs = set()
for glb in glob.glob(os.path.join(output_dir, "glb", "*.glb")):
    # extract word_id from avatar_word_id.glb
    basename = os.path.basename(glb)
    if basename.startswith("avatar_") and basename.endswith(".glb"):
        word_id = basename[len("avatar_"):-4]
        existing_glbs.add(word_id)

def sanitize_word_id(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    for sfx in ["_raw", "_authentic", "_trimmed"]:
        if base.endswith(sfx):
            base = base[:-len(sfx)]
    return base.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("'", "")

# Recursive search for mp4 and mov files
videos = []
for ext in ('*.mp4', '*.mov', '*.MP4', '*.MOV'):
    videos.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

# Remove duplicates if any (due to case-insensitivity on Windows, just in case)
videos = list(set(videos))

print(f"Found {len(videos)} videos in total across all subdirectories.")

videos_to_process = []
for v in videos:
    word_id = sanitize_word_id(v)
    if word_id in existing_glbs:
        print(f"Skipping '{word_id}' - already exists")
    else:
        videos_to_process.append(v)

print(f"\nVideos left to process: {len(videos_to_process)}")

for v in videos_to_process:
    word_id = sanitize_word_id(v)
    print(f"\n=============================================")
    print(f"---> Processing '{word_id}' ({os.path.basename(v)})")
    print(f"=============================================")
    # Call run_pipeline.py for this specific file
    cmd = [
        "python", "run_pipeline.py",
        "--file", v,
        "--output_dir", output_dir,
        "--bake_glb"
    ]
    subprocess.run(cmd, check=False)
