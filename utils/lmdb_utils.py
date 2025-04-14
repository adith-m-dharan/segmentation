import os
import lmdb
import cv2
from tqdm import tqdm
from glob import glob

def basename_no_ext(path):
    """Return the basename without extension."""
    return os.path.splitext(os.path.basename(path))[0]

def save_filtered_to_lmdb(input_dir, lmdb_path, file_list, ext=".png", map_size=1e12):
    """
    Save images (or masks) in input_dir with names in file_list into an LMDB.
    The LMDB key will be the basename without extension.
    """
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(map_size))

    with env.begin(write=True) as txn:
        for file_name in tqdm(sorted(file_list), desc=f"Saving to {os.path.basename(lmdb_path)}"):
            path = os.path.join(input_dir, file_name)
            img = cv2.imread(path)
            if img is None:
                print(f"[Warning] Could not read {path}")
                continue
            # Encode using the given extension (".png")
            _, encoded = cv2.imencode(ext, img)
            key = os.path.splitext(file_name)[0]  # Use basename without extension as key
            txn.put(key.encode("ascii"), encoded.tobytes())

    env.close()

def prepare_lmdb(base_input, base_output):
    """
    Checks for LMDB files for each split (train, test). If they do not exist,
    it builds them by matching files in the images and masks directories based on basename.
    """
    os.makedirs(base_output, exist_ok=True)
    splits = ["train", "test"]

    for split in splits:
        img_dir = os.path.join(base_input, split, "images")
        mask_dir = os.path.join(base_input, split, "masks")
        img_lmdb = os.path.join(base_output, f"{split}_images.lmdb")
        mask_lmdb = os.path.join(base_output, f"{split}_masks.lmdb")

        # Get list of image and mask files
        image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))  # Masks are expected as PNG

        # Build dictionaries with key = basename without extension, value = full filename
        image_dict = {basename_no_ext(p): os.path.basename(p) for p in image_paths}
        mask_dict  = {basename_no_ext(p): os.path.basename(p) for p in mask_paths}

        # Find common basenames (these are the matching keys)
        common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
        matched_image_files = [image_dict[k] for k in common_keys]
        matched_mask_files  = [mask_dict[k] for k in common_keys]

        if not matched_image_files or not matched_mask_files:
            print(f"[Skip] No matching images and masks in {split} — skipping LMDB creation.")
            continue

        if not os.path.exists(img_lmdb):
            print(f"[LMDB] Creating {img_lmdb} with {len(matched_image_files)} matched images...")
            save_filtered_to_lmdb(img_dir, img_lmdb, matched_image_files)
        else:
            print(f"[LMDB] Found {img_lmdb}")

        if not os.path.exists(mask_lmdb):
            print(f"[LMDB] Creating {mask_lmdb} with {len(matched_mask_files)} matched masks...")
            save_filtered_to_lmdb(mask_dir, mask_lmdb, matched_mask_files)
        else:
            print(f"[LMDB] Found {mask_lmdb}")
