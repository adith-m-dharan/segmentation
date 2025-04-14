import os
import shutil
import random
import sys
import requests
import zipfile
import asyncio
import aiohttp
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
from omegaconf import OmegaConf
import argparse

def download_annotation_file(url, output_folder, file_name):
    """
    Downloads a file from the provided URL if it does not exist in the output folder.
    Displays a TQDM progress bar while downloading.
    
    Args:
        url (str): URL to download from.
        output_folder (str): Destination folder.
        file_name (str): The name to save the file under.
    Returns:
        str: Full path to the downloaded file.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, file_name)
    if os.path.exists(file_path):
        print(f"File '{file_name}' already exists in '{output_folder}'.")
        return file_path

    print(f"Downloading file from {url} to {file_path} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error on bad status code

    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192

    with open(file_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading ZIP") as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print("File downloaded successfully.")
    return file_path

def extract_zip(zip_path, extract_to, desired_file):
    """
    Extracts only the desired file from a ZIP archive without preserving its internal folder structure.
    The file is saved directly in the extract_to folder using its basename.
    
    Args:
        zip_path (str): Path of the ZIP file.
        extract_to (str): Directory to save the file.
        desired_file (str): The internal path of the file in the ZIP (e.g. "annotations/instances_train2017.json").
    Returns:
        str: Full path to the extracted file.
    """
    os.makedirs(extract_to, exist_ok=True)
    target_basename = os.path.basename(desired_file)
    extracted_file_path = os.path.join(extract_to, target_basename)
    if os.path.exists(extracted_file_path):
        print(f"Extracted file '{target_basename}' already exists in '{extract_to}'.")
        return extracted_file_path

    print(f"Extracting '{desired_file}' from '{zip_path}' ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for info in zip_ref.infolist():
            if os.path.basename(info.filename) == target_basename:
                file_data = zip_ref.read(info.filename)
                with open(extracted_file_path, 'wb') as f:
                    f.write(file_data)
                break
    print("Extraction complete.")
    return extracted_file_path

# -------- Asynchronous Image Downloading Functions --------

async def download_image_async(session: aiohttp.ClientSession, url: str, out_path: str):
    """
    Downloads a single image asynchronously using aiohttp.
    
    Args:
        session (aiohttp.ClientSession): Active session.
        url (str): Image URL.
        out_path (str): Local file path to save the image.
    """
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(out_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(16384)  # 16KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                print(f"Failed to download {url} (status: {response.status})")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

async def download_images_concurrently(urls, out_paths, max_concurrency: int = 16):
    """
    Downloads multiple images concurrently with a semaphore limiting concurrency.
    Displays one overall progress bar.
    
    Args:
        urls (list): List of image URLs.
        out_paths (list): Corresponding local paths to save the images.
        max_concurrency (int, optional): Maximum concurrent downloads.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url, out_path in zip(urls, out_paths):
            async def task_wrapper(url=url, out_path=out_path):
                async with semaphore:
                    await download_image_async(session, url, out_path)
            tasks.append(task_wrapper())
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading images"):
            await task

def download_images_for_split(coco, img_ids, split_name, output_base, max_concurrency=16):
    """
    For the given image IDs (a split), collects their URLs and output paths,
    and downloads them concurrently.
    
    Args:
        coco (COCO): COCO annotation object.
        img_ids (list): List of image IDs to download.
        split_name (str): Name of the split (train/test/inference).
        output_base (str): Base output directory.
        max_concurrency (int, optional): Maximum concurrent downloads.
    """
    split_dir = os.path.join(output_base, split_name, "images")
    os.makedirs(split_dir, exist_ok=True)
    urls = []
    out_paths = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        dst_path = os.path.join(split_dir, file_name)
        url = img_info.get('coco_url') or img_info.get('flickr_url')
        if url:
            urls.append(url)
            out_paths.append(dst_path)
        else:
            print(f"No download URL for image {img_id} - {file_name}")
    if urls:
        print(f"\nDownloading {len(urls)} images for split '{split_name}'...")
        asyncio.run(download_images_concurrently(urls, out_paths, max_concurrency=max_concurrency))
    else:
        print(f"No images to download for split '{split_name}'.")

def print_class_distribution(coco, img_ids, target_cats, split_name):
    class_counts = {cat_id: 0 for cat_id in target_cats}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        present_cats = set(ann['category_id'] for ann in anns if ann['category_id'] in target_cats)
        for cat_id in present_cats:
            class_counts[cat_id] += 1

    print(f"\nClass-wise image count in '{split_name}' split:")
    for cat_id in target_cats:
        print(f"  Category {cat_id}: {class_counts[cat_id]} images")

# ---------------- Dataset Preparation Function ----------------

def prepare_dataset(coco, max_images, target_cats, output_base, mask_area_threshold, seed, splits):
    """
    Prepares the dataset by filtering images based on COCO annotations,
    downloading images concurrently, and generating segmentation masks.
    
    Args:
        coco (COCO): A loaded COCO instance.
        max_images (int): Maximum number of images to process.
        target_cats (list): List of target category IDs.
        output_base (str): Output directory for dataset splits.
        mask_area_threshold (float): Minimum fraction of image area per category required.
        seed (int, optional): Seed for reproducibility.
    """
    # Create directories for splits.
    for split in ['train', 'test', 'inference']:
        os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, "masks"), exist_ok=True)

    print("Using loaded annotations...")
    all_ids = list(coco.imgs.keys())
    available_ids = all_ids

    cat_to_images = defaultdict(list)

    for img_id in tqdm(available_ids, desc="Finding images per class"):
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        img_area = img_info['width'] * img_info['height']

        for cat_id in target_cats:
            cat_anns = [ann for ann in anns if ann['category_id'] == cat_id]
            total_area = sum(ann['area'] for ann in cat_anns)
            if total_area / img_area >= mask_area_threshold:
                cat_to_images[cat_id].append(img_id)

    # Sample same number of images per category
    min_samples_per_cat = min(len(imgs) for imgs in cat_to_images.values())
    print(f"Sampling {min_samples_per_cat} images per category")

    balanced_ids = set()
    for cat_id in target_cats:
        selected = random.sample(cat_to_images[cat_id], min_samples_per_cat)
        balanced_ids.update(selected)

    # Convert to list and shuffle
    balanced_ids = list(balanced_ids)
    random.shuffle(balanced_ids)

    if max_images is not None and len(balanced_ids) > max_images:
        print(f"Truncating to max_images = {max_images}")
        balanced_ids = balanced_ids[:max_images]

    print(f"Final unique image count after balancing and truncating: {len(balanced_ids)}")

    num_total = len(balanced_ids)
    num_train = int(splits.train * num_total)
    num_test = int(splits.test * num_total)
    num_inference = num_total - num_train - num_test

    train_ids = balanced_ids[:num_train]
    test_ids = balanced_ids[num_train:num_train + num_test]
    inference_ids = balanced_ids[num_train + num_test:]

    # Print per-class stats
    print_class_distribution(coco, train_ids, target_cats, "train")
    print_class_distribution(coco, test_ids, target_cats, "test")
    print_class_distribution(coco, inference_ids, target_cats, "inference")

    # Download images concurrently for each split.
    download_images_for_split(coco, train_ids, "train", output_base)
    download_images_for_split(coco, test_ids, "test", output_base)
    download_images_for_split(coco, inference_ids, "inference", output_base)

    # Generate segmentation masks synchronously.
    def generate_and_save_coco_masks(image_ids, split_name):
        mask_dir = os.path.join(output_base, split_name, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        for img_id in tqdm(image_ids, desc=f"\nGenerating masks for {split_name}"):
            img_info = coco.loadImgs(img_id)[0]
            W, H = img_info['width'], img_info['height']
            file_name = img_info['file_name']
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            mask = np.zeros((H, W), dtype=np.uint8)
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in target_cats:
                    continue
                label = target_cats.index(cat_id) + 1
                segm = ann['segmentation']
                if isinstance(segm, list):
                    for poly in segm:
                        if len(poly) >= 6:
                            poly_np = np.array(poly).reshape(-1, 2)
                            img_mask = Image.new('L', (W, H), 0)
                            ImageDraw.Draw(img_mask).polygon([tuple(p) for p in poly_np], fill=label)
                            mask = np.maximum(mask, np.array(img_mask, dtype=np.uint8))
                elif isinstance(segm, dict) and 'counts' in segm:
                    m = maskUtils.decode(segm)
                    if m.ndim == 3:
                        m = np.any(m, axis=2)
                    mask[m > 0] = label
            mask_path = os.path.join(mask_dir, os.path.basename(file_name).replace('.jpg', '.png'))
            Image.fromarray(mask).save(mask_path)

    generate_and_save_coco_masks(train_ids, "train")
    generate_and_save_coco_masks(test_ids, "test")
    generate_and_save_coco_masks(inference_ids, "inference")

    print("Dataset preparation complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare COCO dataset with config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to YAML configuration file"
    )
    return parser.parse_args()

def load_config(cfg_path):
    return OmegaConf.load(cfg_path)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    annotations_folder = os.path.join(cfg.data_dir, "annotations")
    annotations_url = cfg.annotations.zip_url
    annotations_zip_name = cfg.annotations.zip_name
    desired_annotation_file = cfg.annotations.json_path
    desired_local_annotation_file = os.path.join(annotations_folder, os.path.basename(desired_annotation_file))
    
    # Check if the annotation JSON exists.
    if os.path.exists(desired_local_annotation_file):
        print(f"Annotation file '{desired_local_annotation_file}' already exists. Skipping ZIP download and extraction.")
        try:
            coco_instance = COCO(desired_local_annotation_file)
            print("Successfully loaded annotation JSON.")
        except Exception as e:
            print(f"Error loading annotation file: {e}. Redownloading and extracting.")
            os.remove(desired_local_annotation_file)
            coco_instance = None
        # If ZIP file exists, delete it.
        zip_path = os.path.join(annotations_folder, annotations_zip_name)
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
                print(f"Deleted extraneous ZIP file: {zip_path}")
            except Exception as e:
                print(f"Error deleting ZIP file: {e}")
    else:
        coco_instance = None

    if coco_instance is None:
        zip_path = os.path.join(annotations_folder, annotations_zip_name)
        if os.path.exists(zip_path):
            print(f"ZIP file '{zip_path}' already exists. Attempting extraction...")
            try:
                annotation_file = extract_zip(zip_path, annotations_folder, desired_annotation_file)
            except Exception as e:
                print(f"Error extracting ZIP: {e}. Deleting ZIP and downloading again.")
                os.remove(zip_path)
                zip_path = download_annotation_file(annotations_url, annotations_folder, annotations_zip_name)
                annotation_file = extract_zip(zip_path, annotations_folder, desired_annotation_file)
        else:
            zip_path = download_annotation_file(annotations_url, annotations_folder, annotations_zip_name)
            annotation_file = extract_zip(zip_path, annotations_folder, desired_annotation_file)
        try:
            coco_instance = COCO(annotation_file)
            print("Successfully loaded annotation JSON after extraction.")
        except Exception as e:
            print(f"Error loading extracted annotation file: {e}. Exiting.")
            sys.exit(1)
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
                print(f"Deleted ZIP file: {zip_path}")
            except Exception as e:
                print(f"Error deleting ZIP file: {e}")

    output_base = os.path.join(cfg.data_dir, "prepared")
    max_images = cfg.max_images
    target_categories = cfg.target_categories
    mask_area_threshold = cfg.mask_area_threshold
    seed = cfg.seed
    splits = cfg.splits

    prepare_dataset(
        coco_instance,
        max_images,
        target_categories,
        output_base,
        mask_area_threshold,
        seed=seed,
        splits=splits
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(0)