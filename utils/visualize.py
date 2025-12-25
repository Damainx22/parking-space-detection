import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import random
from PIL import Image


def visualize_dataset_samples(data_dir, num_samples=8, save_path=None):
    """
    Visualize random samples from the dataset

    Args:
        data_dir: Path to dataset directory
        num_samples: Number of samples to show (must be even)
        save_path: Optional path to save the figure
    """
    data_dir = Path(data_dir)

    # Get sample images
    empty_images = list((data_dir / 'train' / 'empty').glob('*.jpg'))
    occupied_images = list((data_dir / 'train' / 'occupied').glob('*.jpg'))

    if len(empty_images) == 0 or len(occupied_images) == 0:
        print("ERROR: No images found in dataset")
        return

    # Sample equal number from each class
    samples_per_class = num_samples // 2
    empty_samples = random.sample(empty_images, min(samples_per_class, len(empty_images)))
    occupied_samples = random.sample(occupied_images, min(samples_per_class, len(occupied_images)))

    # Create figure
    fig, axes = plt.subplots(2, samples_per_class, figsize=(15, 6))
    fig.suptitle('Dataset Samples: Empty vs Occupied Parking Spaces', fontsize=14, fontweight='bold')

    # Plot empty spaces
    for i, img_path in enumerate(empty_samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Empty', fontweight='bold')

    # Plot occupied spaces
    for i, img_path in enumerate(occupied_samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Occupied', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Dataset samples saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_dataset_distribution(data_dir, save_path=None):
    """
    Plot class distribution across train/val/test splits

    Args:
        data_dir: Path to dataset directory
        save_path: Optional path to save the figure
    """
    data_dir = Path(data_dir)

    splits = ['train', 'val', 'test']
    classes = ['empty', 'occupied']

    # Count samples
    counts = {}
    for split in splits:
        counts[split] = {}
        for cls in classes:
            cls_dir = data_dir / split / cls
            if cls_dir.exists():
                counts[split][cls] = len(list(cls_dir.glob('*.jpg')))
            else:
                counts[split][cls] = 0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Stacked bar chart
    empty_counts = [counts[split]['empty'] for split in splits]
    occupied_counts = [counts[split]['occupied'] for split in splits]

    x = np.arange(len(splits))
    width = 0.6

    axes[0].bar(x, empty_counts, width, label='Empty', color='green', alpha=0.7)
    axes[0].bar(x, occupied_counts, width, bottom=empty_counts, label='Occupied', color='red', alpha=0.7)

    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Dataset Distribution by Split')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Add count labels
    for i, split in enumerate(splits):
        total = empty_counts[i] + occupied_counts[i]
        axes[0].text(i, total + max(total * 0.02, 10), str(total),
                    ha='center', va='bottom', fontweight='bold')

    # Plot 2: Class distribution pie chart (total)
    total_empty = sum(empty_counts)
    total_occupied = sum(occupied_counts)

    axes[1].pie([total_empty, total_occupied],
               labels=['Empty', 'Occupied'],
               autopct='%1.1f%%',
               colors=['green', 'red'],
               alpha=0.7,
               startangle=90)
    axes[1].set_title(f'Overall Class Distribution\n(Total: {total_empty + total_occupied} samples)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    print("\nDataset Statistics:")
    print("=" * 60)
    for split in splits:
        empty = counts[split]['empty']
        occupied = counts[split]['occupied']
        total = empty + occupied
        if total > 0:
            print(f"{split.capitalize():6} | Empty: {empty:6} | Occupied: {occupied:6} | Total: {total:6}")
    print("=" * 60)
    print(f"{'Total':6} | Empty: {total_empty:6} | Occupied: {total_occupied:6} | Total: {total_empty + total_occupied:6}")
    print("=" * 60)


def visualize_model_predictions(image_paths, predictions, labels, save_path=None, num_samples=16):
    """
    Visualize model predictions with ground truth

    Args:
        image_paths: List of image paths
        predictions: List of predicted classes
        labels: List of true labels
        save_path: Optional path to save the figure
        num_samples: Number of samples to visualize
    """
    class_names = ['Empty', 'Occupied']

    # Randomly sample images
    if len(image_paths) > num_samples:
        indices = random.sample(range(len(image_paths)), num_samples)
    else:
        indices = range(len(image_paths))

    # Calculate grid size
    cols = 4
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for idx, i in enumerate(indices):
        img = cv2.imread(str(image_paths[i]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = predictions[i]
        true = labels[i]
        correct = pred == true

        # Determine border color
        color = 'green' if correct else 'red'

        axes[idx].imshow(img)
        axes[idx].axis('off')

        # Title with prediction and ground truth
        title = f"Pred: {class_names[pred]}\nTrue: {class_names[true]}"
        axes[idx].set_title(title, color=color, fontweight='bold')

        # Add colored border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    # Hide empty subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Model Predictions (Green=Correct, Red=Incorrect)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Predictions visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_parking_lot_overlay(parking_lot_image, space_coords, predictions, save_path=None):
    """
    Overlay parking space predictions on full parking lot image

    Args:
        parking_lot_image: Path to full parking lot image or numpy array
        space_coords: List of parking space coordinates [(x1,y1,x2,y2), ...]
        predictions: List of predictions (0=empty, 1=occupied) for each space
        save_path: Optional path to save the result
    """
    # Load image
    if isinstance(parking_lot_image, (str, Path)):
        image = cv2.imread(str(parking_lot_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = parking_lot_image.copy()

    # Draw boxes for each parking space
    for coords, pred in zip(space_coords, predictions):
        x1, y1, x2, y2 = coords

        # Choose color based on prediction
        color = (0, 255, 0) if pred == 0 else (255, 0, 0)  # Green for empty, Red for occupied
        label = "Empty" if pred == 0 else "Occupied"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add label
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Parking Lot Occupancy Detection')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Parking lot overlay saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    return image


if __name__ == "__main__":
    # Example usage
    print("Visualization Utilities")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - visualize_dataset_samples()")
    print("  - plot_dataset_distribution()")
    print("  - visualize_model_predictions()")
    print("  - visualize_parking_lot_overlay()")
    print("\nUsage:")
    print("  from utils.visualize import *")
    print("  visualize_dataset_samples('data/processed', save_path='results/samples.png')")
