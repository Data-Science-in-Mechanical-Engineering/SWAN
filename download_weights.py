import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download, HfApi

# Directories
COMFYUI_MODELS_DIR = Path("./comfyui_models")
WEIGHTS_DIR = Path("./weights")

# Grouped dictionary for better organization and reporting
MODELS_TO_DOWNLOAD = {
    "Core Vision Models": [
        # (Repo ID, Path inside HF repo, Target Directory, Is Snapshot?)
        ("facebook/cotracker3", "baseline_offline.pth", WEIGHTS_DIR / "CoTracker3", False),
        ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", WEIGHTS_DIR / "sam2.1-hiera-large", False),
        ("IDEA-Research/grounding-dino-base", None, WEIGHTS_DIR / "grounding-dino-base", True),
    ],
    "Z-Image Turbo": [
        ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", COMFYUI_MODELS_DIR / "text_encoders", False),
        ("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors", COMFYUI_MODELS_DIR / "diffusion_models", False),
        ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", COMFYUI_MODELS_DIR / "vae", False),
    ],
    "Wan 2.2": [
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", COMFYUI_MODELS_DIR / "vae", False),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", COMFYUI_MODELS_DIR / "text_encoders", False),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", COMFYUI_MODELS_DIR / "diffusion_models", False),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", COMFYUI_MODELS_DIR / "diffusion_models", False),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", COMFYUI_MODELS_DIR / "loras", False),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", COMFYUI_MODELS_DIR / "loras", False),
    ],
    "Qwen3.5-9B": [
        ("Qwen/Qwen3.5-9B", "None", WEIGHTS_DIR / "Qwen3.5-9B", True),
    ]
}

def format_size(size_bytes: int) -> str:
    """Converts bytes to a human-readable format."""
    if not size_bytes:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_snapshot_size(api: HfApi, repo_id: str) -> int:
    """Calculates total size of all files in a repository for snapshot downloads."""
    try:
        info = api.model_info(repo_id, files_metadata=True)
        return sum(sibling.size for sibling in info.siblings if sibling.size is not None)
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch metadata for {repo_id} ({e})")
        return 0

def print_installation_report() -> tuple[int, int]:
    """Fetches metadata and prints a size report, returning missing sizes for core and video."""
    print("📊 Fetching model sizes from Hugging Face...\n")
    api = HfApi()

    # Flatten paths by repo to minimize API calls (excluding snapshots for now)
    repo_files = {}
    for group_models in MODELS_TO_DOWNLOAD.values():
        for repo_id, hf_path, _, is_snapshot in group_models:
            if not is_snapshot:
                repo_files.setdefault(repo_id, []).append(hf_path)

    # Fetch sizes via API
    file_sizes = {}
    for repo, paths in repo_files.items():
        try:
            info = api.model_info(repo, files_metadata=True)
            for sibling in info.siblings:
                if sibling.rfilename in paths:
                    file_sizes[(repo, sibling.rfilename)] = sibling.size
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch metadata for {repo} ({e})")

    print("=" * 90)
    print(f"{'Model File / Repo':<50} | {'Target Dir':<16} | {'Size':<15}")
    print("=" * 90)

    total_size_core = 0
    missing_size_core = 0
    total_size_video = 0
    missing_size_video = 0

    for group_name, models in MODELS_TO_DOWNLOAD.items():
        print(f"\n📁 [ {group_name} ]")
        print("-" * 90)
        
        group_total = 0
        group_missing = 0
        is_video_group = group_name in ["Z-Image Turbo", "Wan 2.2"]

        for repo_id, hf_path, target_dir, is_snapshot in models:
            if is_snapshot:
                filename = f"Snapshot: {repo_id.split('/')[-1]}"
                size = get_snapshot_size(api, repo_id)
                
                # Simple check: if the dir exists and isn't empty, assume downloaded
                if target_dir.exists() and any(target_dir.iterdir()):
                     status = f"✅ Local (~{format_size(size)})"
                else:
                     status = format_size(size)
                     group_missing += size
                     if is_video_group: missing_size_video += size
                     else: missing_size_core += size
            else:
                filename = Path(hf_path).name
                size = file_sizes.get((repo_id, hf_path), 0)
                target_file = target_dir / filename
                
                if target_file.exists():
                    status = f"✅ Local ({format_size(size)})"
                else:
                    status = format_size(size)
                    group_missing += size
                    if is_video_group: missing_size_video += size
                    else: missing_size_core += size

            group_total += size
            if is_video_group: total_size_video += size
            else: total_size_core += size
            
            # Formatting the target directory string for display
            display_dir = str(target_dir).replace('comfyui_models', 'comfy_models')
            if len(display_dir) > 30:
                 display_dir = "..." + display_dir[-27:]
            
            print(f"{filename:<50} | {display_dir:<30} | {status:<15}")
        
        # Print Group Subtotals
        print("-" * 90)
        print(f"{group_name} Total: {format_size(group_total)} "
              f"(Missing: {format_size(group_missing)})")

    print("=" * 90)
    print("Summary:")
    print(f"Core Pipeline Missing Data:  {format_size(missing_size_core)}")
    print(f"Video Pipeline Missing Data: {format_size(missing_size_video)}")
    print(f"Total Missing Data:          {format_size(missing_size_core + missing_size_video)}\n")
    
    return missing_size_core, missing_size_video

def main():
    missing_core, missing_video = print_installation_report()
    
    if missing_core == 0 and missing_video == 0:
        print("🎉 All models are already installed locally! Exiting.")
        return

    print("Options:")
    print("  1: Download Core Vision Models ONLY")
    print("  2: Download ALL Models (Core + Video)")
    print("  0: Cancel")
    
    choice = input("\nEnter your choice (0, 1, or 2): ").strip()
    
    if choice not in ['1', '2']:
        print("🛑 Download cancelled or invalid input.")
        return

    download_video = (choice == '2')

    for group_name, models in MODELS_TO_DOWNLOAD.items():
        is_video_group = group_name in ["Z-Image Turbo", "Wan 2.2"]
        
        if is_video_group and not download_video:
            continue
            
        print(f"\n📦 Processing {group_name}...")
        
        for repo_id, hf_path, target_dir, is_snapshot in models:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if is_snapshot:
                # We do a basic check to skip existing snapshots
                if target_dir.exists() and any(target_dir.iterdir()):
                     continue
                
                print(f"   ⏳ Downloading snapshot of {repo_id}...")
                try:
                    # snapshot_download automatically handles copying to a target dir
                    snapshot_download(repo_id=repo_id, local_dir=target_dir)
                    print(f"   ✅ Successfully installed {repo_id}")
                except Exception as e:
                    print(f"   ❌ Failed to download {repo_id}. Error: {e}")
                    
            else:
                filename = Path(hf_path).name
                target_file = target_dir / filename

                if target_file.exists():
                    continue

                print(f"   ⏳ Downloading {filename}...")
                try:
                    cached_file = hf_hub_download(repo_id=repo_id, filename=hf_path)
                    print(f"      Copying to {target_dir.name}/{filename}...")
                    shutil.copyfile(cached_file, target_file)
                    print(f"   ✅ Successfully installed {filename}")
                except Exception as e:
                    print(f"   ❌ Failed to download {filename}. Error: {e}")

    print("\n🎉 Required models are ready!")

if __name__ == "__main__":
    main()