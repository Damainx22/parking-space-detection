"""
Process PKLot COCO format dataset and extract parking space patches
Organizes into train/val/test splits with empty/occupied classes
"""

import json
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil


def process_coco_split(coco_json_path, images_dir, output_dir, split_name):
    """
    Process one split (train/val/test) of COCO format PKLot dataset

    Args:
        coco_json_path: Path to _annotations.coco.json
        images_dir: Directory containing the images
        output_dir: Output directory for processed data
        split_name: Name of split ('train', 'val', or 'test')
    """
    print(f"\nProcessing {split_name} split...")

    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    for class_name in ['empty', 'occupied']:
        (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"  Categories found: {categories}")

    # Create image ID to filename mapping
    images = {img['id']: img['file_name'] for img in coco_data['images']}

    # Process annotations
    stats = {'empty': 0, 'occupied': 0}

    for ann in tqdm(coco_data['annotations'], desc=f"  Processing {split_name}"):
        # Get image
        image_id = ann['image_id']
        image_filename = images[image_id]
        image_path = images_dir / image_filename

        if not image_path.exists():
            continue

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        # Get bounding box
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = [int(v) for v in bbox]

        # Crop parking space
        crop = img[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        # Resize to standard size
        crop_resized = cv2.resize(crop, (64, 64))

        # Get category
        category_id = ann['category_id']
        category_name = categories[category_id].lower()

        # Map category to our classes
        if 'empty' in category_name or category_name == 'space-empty':
            class_name = 'empty'
        elif 'occupied' in category_name or category_name == 'space-occupied':
            class_name = 'occupied'
        else:
            print(f"Unknown category: {category_name}, skipping...")
            continue

        # Save patch
        image_stem = Path(image_filename).stem
        output_path = output_dir / split_name / class_name / f"{image_stem}_ann{ann['id']}.jpg"
        cv2.imwrite(str(output_path), crop_resized)

        stats[class_name] += 1

    print(f"  {split_name.capitalize()} - Empty: {stats['empty']}, Occupied: {stats['occupied']}")
    return stats


def main():
    print("=" * 70)
    print("Processing PKLot COCO Dataset")
    print("=" * 70)

    # Paths
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    # Process each split
    total_stats = {}

    for split in ['train', 'valid', 'test']:
        split_dir = raw_dir / split
        coco_json = split_dir / "_annotations.coco.json"

        if not coco_json.exists():
            print(f"\nWarning: {coco_json} not found, skipping {split}")
            continue

        # Map 'valid' to 'val'
        output_split = 'val' if split == 'valid' else split

        stats = process_coco_split(
            coco_json_path=coco_json,
            images_dir=split_dir,
            output_dir=processed_dir,
            split_name=output_split
        )

        total_stats[output_split] = stats

    # Print summary
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)

    print("\nDataset Statistics:")
    print("-" * 70)
    for split, stats in total_stats.items():
        total = stats['empty'] + stats['occupied']
        print(f"{split.capitalize():6} | Empty: {stats['empty']:6} | Occupied: {stats['occupied']:6} | Total: {total:6}")

    total_empty = sum(s['empty'] for s in total_stats.values())
    total_occupied = sum(s['occupied'] for s in total_stats.values())
    total_all = total_empty + total_occupied

    print("-" * 70)
    print(f"{'Total':6} | Empty: {total_empty:6} | Occupied: {total_occupied:6} | Total: {total_all:6}")
    print("=" * 70)

    print(f"\nProcessed dataset saved to: {processed_dir}")
    print("\nNext steps:")
    print("  1. Run: python example.py")
    print("  2. Then: python train.py")


if __name__ == "__main__":
    main()
