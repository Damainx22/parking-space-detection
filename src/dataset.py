import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path


class ParkingSpaceDataset(Dataset):
    """
    Dataset for parking space classification (Occupied vs Empty)
    Works with pre-cropped parking space images or extracts spaces from full images.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to parking space images
            labels: List of labels (0 = empty, 1 = occupied)
            transform: Optional torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        assert len(self.image_paths) == len(self.labels), \
            "Number of images and labels must match"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


class PKLotDatasetExtractor:
    """
    Helper class to extract parking space patches from PKLot dataset
    PKLot structure: images with XML files containing space coordinates
    """
    def __init__(self, dataset_path, output_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def parse_xml(self, xml_path):
        """Parse PKLot XML file to extract parking space coordinates and labels"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        spaces = []
        for space in root.findall('.//space'):
            space_id = space.get('id')
            occupied = space.get('occupied') == '1'

            # Get coordinates
            contour = space.find('contour')
            if contour is not None:
                points = []
                for point in contour.findall('point'):
                    x = int(point.get('x'))
                    y = int(point.get('y'))
                    points.append([x, y])

                spaces.append({
                    'id': space_id,
                    'occupied': occupied,
                    'points': np.array(points)
                })

        return spaces

    def extract_space_patch(self, image, points, margin=5):
        """Extract parking space region from image using polygon points"""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points)

        # Add margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + 2 * margin
        h = h + 2 * margin

        # Crop the region
        patch = image[y:y+h, x:x+w]

        return patch

    def process_dataset(self, image_size=(64, 64)):
        """
        Process entire PKLot dataset and extract parking space patches
        Saves patches organized by class: output_path/train/occupied, output_path/train/empty, etc.
        """
        print("Processing PKLot dataset...")

        # Create output directories
        for split in ['train', 'val', 'test']:
            for label in ['empty', 'occupied']:
                (self.output_path / split / label).mkdir(parents=True, exist_ok=True)

        # Find all XML files in dataset
        xml_files = list(self.dataset_path.rglob('*.xml'))

        if len(xml_files) == 0:
            print(f"No XML files found in {self.dataset_path}")
            print("Please ensure PKLot dataset is properly extracted.")
            return

        print(f"Found {len(xml_files)} XML files")

        total_spaces = 0
        occupied_count = 0
        empty_count = 0

        for idx, xml_path in enumerate(xml_files):
            # Find corresponding image
            img_path = xml_path.with_suffix('.jpg')
            if not img_path.exists():
                continue

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Parse XML to get space coordinates
            spaces = self.parse_xml(xml_path)

            # Determine split (80% train, 10% val, 10% test)
            if idx % 10 == 8:
                split = 'val'
            elif idx % 10 == 9:
                split = 'test'
            else:
                split = 'train'

            # Extract each parking space
            for space in spaces:
                # Extract patch
                patch = self.extract_space_patch(image, space['points'])

                if patch.size == 0:
                    continue

                # Resize to standard size
                patch_resized = cv2.resize(patch, image_size)

                # Determine label directory
                label_dir = 'occupied' if space['occupied'] else 'empty'

                # Save patch
                output_file = self.output_path / split / label_dir / \
                              f"{xml_path.stem}_{space['id']}.jpg"
                cv2.imwrite(str(output_file), patch_resized)

                total_spaces += 1
                if space['occupied']:
                    occupied_count += 1
                else:
                    empty_count += 1

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(xml_files)} images, "
                      f"extracted {total_spaces} spaces")

        print(f"\nDataset processing complete!")
        print(f"Total spaces extracted: {total_spaces}")
        print(f"Occupied: {occupied_count}, Empty: {empty_count}")
        print(f"Class distribution: {occupied_count/total_spaces:.2%} occupied")


def load_simple_dataset(data_dir, image_size=(64, 64), batch_size=32):
    """
    Load pre-organized dataset from directory structure:
    data_dir/
        train/
            empty/
            occupied/
        val/
            empty/
            occupied/
        test/
            empty/
            occupied/

    Returns train, val, test DataLoaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    data_dir = Path(data_dir)

    # Helper function to load split
    def load_split(split_name, transform):
        split_dir = data_dir / split_name
        if not split_dir.exists():
            return None

        image_paths = []
        labels = []

        # Load empty spaces (label 0)
        empty_dir = split_dir / 'empty'
        if empty_dir.exists():
            for img_path in empty_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(0)

        # Load occupied spaces (label 1)
        occupied_dir = split_dir / 'occupied'
        if occupied_dir.exists():
            for img_path in occupied_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(1)

        if len(image_paths) == 0:
            return None

        dataset = ParkingSpaceDataset(image_paths, labels, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split_name=='train'),
                          num_workers=0, pin_memory=True)

        return loader

    # Load all splits
    train_loader = load_split('train', train_transform)
    val_loader = load_split('val', test_transform)
    test_loader = load_split('test', test_transform)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage: Extract PKLot dataset
    print("PKLot Dataset Extractor")
    print("-" * 50)
    print("To use this script:")
    print("1. Download PKLot dataset from: https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz")
    print("2. Extract the dataset")
    print("3. Run this script with proper paths")
    print()

    # Example extraction (update paths as needed)
    # extractor = PKLotDatasetExtractor(
    #     dataset_path='data/raw/PKLot',
    #     output_path='data/processed'
    # )
    # extractor.process_dataset(image_size=(64, 64))
