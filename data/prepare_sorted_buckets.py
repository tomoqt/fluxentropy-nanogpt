import os
import subprocess
import glob

# Clone the dataset if it doesn't exist
dataset_path = "fineweb10B-gpt2-fluxentropy"
if not os.path.exists(dataset_path):
    print("Cloning dataset repository...")
    subprocess.run(["git", "clone", "https://huggingface.co/datasets/baslak/fineweb10B-gpt2-fluxentropy"], check=True)
    print("Dataset cloned successfully.")

# Always ensure we have the LFS files
print("Fetching LFS files...")
subprocess.run(["git", "lfs", "fetch", "--all"], cwd=dataset_path, check=True)
print("Checking out LFS files...")
subprocess.run(["git", "lfs", "checkout"], cwd=dataset_path, check=True)

# Parameters
num_folders = 5    # We have 5 folders (1-5)
num_buckets = 6    # Each folder has 6 buckets (1-6)

base_dir = "data"
sorted_dir = "sorted"

# Create necessary directories
os.makedirs(base_dir, exist_ok=True)
os.makedirs(sorted_dir, exist_ok=True)

# For each bucket index (1-6)
for k in range(1, num_buckets + 1):
    print(f"Processing bucket_{k}.bin files...")
    
    # Create the combined bucket file
    combined_filename = f"bucket_{k}.bin"
    combined_path = os.path.join(sorted_dir, combined_filename)
    
    with open(combined_path, "wb") as outfile:
        # For each folder (1-5)
        for folder in range(1, num_folders + 1):
            folder_name = f"fineweb_train_{folder:06d}"
            bucket_path = os.path.join(dataset_path, folder_name, f"bucket_{k}.bin")
            print(f"  Merging {bucket_path}")
            
            if os.path.exists(bucket_path):
                # Use git-lfs to get the actual content
                result = subprocess.run(
                    ["git", "lfs", "smudge"],
                    input=open(bucket_path, 'rb').read(),
                    capture_output=True,
                    cwd=dataset_path
                )
                if result.returncode == 0:
                    outfile.write(result.stdout)
                else:
                    print(f"  Error processing {bucket_path}: {result.stderr.decode()}")
            else:
                print(f"  Warning: {bucket_path} not found")

print("All buckets have been combined.")
