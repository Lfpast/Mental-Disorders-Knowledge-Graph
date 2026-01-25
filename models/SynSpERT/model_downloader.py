import os
import subprocess
import shutil
import sys

# Define constants
# Assuming this script is located in models/SynSpERT/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "InputsAndOutputs", "pretrained"))
MODEL_URL = "https://huggingface.co/GanjinZero/coder_eng_pp"
MODEL_DIR_NAME = "coder_eng_pp" # The default folder name git clone creates

def run_command(command, cwd=None):
    """Run a shell command."""
    print(f"[EXEC] {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        sys.exit(1)

def main():
    print("="*50)
    print("[INFO] Starting Model Download Pipeline")
    print("="*50)

    # 1. Ensure target directory exists
    if not os.path.exists(TARGET_DIR):
        print(f"[INFO] Creating directory: {TARGET_DIR}")
        os.makedirs(TARGET_DIR)
    else:
        print(f"[INFO] Target directory exists: {TARGET_DIR}")

    # 2. Check for git-lfs (recommended for HF models)
    try:
        subprocess.run("git lfs --version", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[INFO] git-lfs is installed.")
    except subprocess.CalledProcessError:
        print("[WARN] git-lfs not found. Download might be slow or incomplete for large files.")
        print("[WARN] Please install git-lfs: https://git-lfs.com/")
        # We continue anyway, as git clone might still work for pointers or small models

    # 3. Clone the repository
    # We clone into the TARGET_DIR/coder_eng_pp temporarily to avoid cluttering potentially populated dirs
    clone_target_path = os.path.join(TARGET_DIR, MODEL_DIR_NAME)
    
    if os.path.exists(clone_target_path):
        print(f"[INFO] Removing existing temporary clone directory: {clone_target_path}")
        shutil.rmtree(clone_target_path)

    print(f"[INFO] Cloning {MODEL_URL} into {clone_target_path}...")
    run_command(f"git clone {MODEL_URL} {MODEL_DIR_NAME}", cwd=TARGET_DIR)

    # 4. Move files to TARGET_DIR and flatten
    print(f"[INFO] Moving files from {clone_target_path} to {TARGET_DIR}...")
    
    files_moved = 0
    for item in os.listdir(clone_target_path):
        src_path = os.path.join(clone_target_path, item)
        dst_path = os.path.join(TARGET_DIR, item)

        # Skip moving .git to avoid making 'pretrained' a nested git repo if desired, 
        # or keep it if we want to track the model version. 
        # Usually for deployment/usage, .git is not strictly needed.
        # Let's move everything including hidden files to be safe with the user's request "all files".
        
        if os.path.exists(dst_path):
            print(f"[WARN] Overwriting existing file/directory: {item}")
            if os.path.isdir(dst_path):
                shutil.rmtree(dst_path)
            else:
                os.remove(dst_path)
        
        shutil.move(src_path, dst_path)
        files_moved += 1

    print(f"[INFO] Moved {files_moved} items.")

    # 5. Remove the empty cloned directory
    print(f"[INFO] Removing empty directory: {clone_target_path}")
    if os.path.exists(clone_target_path):
         shutil.rmtree(clone_target_path)
    
    print("="*50)
    print("[INFO] Model download and setup complete.")
    print("="*50)

if __name__ == "__main__":
    main()
