"""
Example script demonstrating parking space detection usage
Run this after preparing your dataset
"""

import torch
from pathlib import Path


def check_environment():
    """Check if environment is set up correctly"""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    # Check Python packages
    try:
        import cv2
        print("[OK] OpenCV installed")
    except ImportError:
        print("[X] OpenCV not installed - run: pip install opencv-python")

    try:
        import torchvision
        print("[OK] Torchvision installed")
    except ImportError:
        print("[X] Torchvision not installed - run: pip install torchvision")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA available - GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[X] CUDA not available - will use CPU (slower)")

    # Check dataset
    data_dir = Path('data/processed')
    if (data_dir / 'train' / 'empty').exists():
        num_train = len(list((data_dir / 'train' / 'empty').glob('*.jpg')))
        num_train += len(list((data_dir / 'train' / 'occupied').glob('*.jpg')))
        print(f"[OK] Dataset found - {num_train} training images")
    else:
        print("[X] Dataset not found - prepare dataset first")

    print("=" * 70)


def test_model_creation():
    """Test creating different models"""
    print("\n" + "=" * 70)
    print("Testing Model Creation")
    print("=" * 70)

    from src.model import get_model, count_parameters

    models = ['simple_cnn', 'resnet18', 'mobilenet']

    for model_name in models:
        try:
            model = get_model(model_name, num_classes=2, pretrained=False)
            params = count_parameters(model)
            print(f"[OK] {model_name:15} - {params:>10,} parameters")
        except Exception as e:
            print(f"[X] {model_name:15} - Error: {e}")

    print("=" * 70)


def test_dataset_loading():
    """Test loading dataset"""
    print("\n" + "=" * 70)
    print("Testing Dataset Loading")
    print("=" * 70)

    try:
        from src.dataset import load_simple_dataset

        train_loader, val_loader, test_loader = load_simple_dataset(
            'data/processed',
            image_size=(64, 64),
            batch_size=4
        )

        if train_loader:
            print(f"[OK] Training set loaded - {len(train_loader.dataset)} samples")
            print(f"[OK] Validation set loaded - {len(val_loader.dataset)} samples")
            print(f"[OK] Test set loaded - {len(test_loader.dataset)} samples")

            # Test loading a batch
            images, labels = next(iter(train_loader))
            print(f"[OK] Batch loaded - shape: {images.shape}, labels: {labels.shape}")
        else:
            print("[X] Could not load dataset")
            print("   Make sure images are in data/processed/train/empty/ and data/processed/train/occupied/")

    except Exception as e:
        print(f"[X] Error loading dataset: {e}")

    print("=" * 70)


def visualize_samples():
    """Visualize dataset samples"""
    print("\n" + "=" * 70)
    print("Visualizing Dataset Samples")
    print("=" * 70)

    try:
        from utils.visualize import visualize_dataset_samples, plot_dataset_distribution

        # Create results directory
        Path('results').mkdir(exist_ok=True)

        # Visualize samples
        print("Creating sample visualization...")
        visualize_dataset_samples('data/processed', num_samples=8,
                                 save_path='results/dataset_samples.png')
        print("[OK] Saved to: results/dataset_samples.png")

        # Plot distribution
        print("\nCreating distribution plot...")
        plot_dataset_distribution('data/processed',
                                save_path='results/dataset_distribution.png')
        print("[OK] Saved to: results/dataset_distribution.png")

    except Exception as e:
        print(f"[X] Error during visualization: {e}")

    print("=" * 70)


def quick_training_test():
    """Run a quick training test (3 epochs)"""
    print("\n" + "=" * 70)
    print("Quick Training Test (3 epochs)")
    print("=" * 70)
    print("This will train a SimpleCNN for 3 epochs as a test")
    response = input("Continue? (y/n): ")

    if response.lower() != 'y':
        print("Skipped.")
        return

    try:
        from src.model import get_model
        from src.dataset import load_simple_dataset
        from train import Trainer

        # Load small dataset
        train_loader, val_loader, _ = load_simple_dataset(
            'data/processed',
            image_size=(64, 64),
            batch_size=16
        )

        if not train_loader:
            print("[X] Could not load dataset")
            return

        # Create simple model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = get_model('simple_cnn', num_classes=2)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=0.001,
            checkpoint_dir='test_checkpoints'
        )

        # Train for 3 epochs
        print("\nStarting training...")
        trainer.train(num_epochs=3, early_stopping_patience=5)

        print("\n[OK] Training test completed successfully!")
        print("  Check test_checkpoints/ for results")

    except Exception as e:
        print(f"[X] Error during training: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)


def main():
    """Run all example tests"""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "Parking Space Detection - Examples")
    print("=" * 70)

    # Run checks
    check_environment()
    test_model_creation()
    test_dataset_loading()

    # Visualization (if dataset exists)
    if Path('data/processed/train').exists():
        visualize_samples()

    # Optional: Quick training test
    print("\n" + "=" * 70)
    print("All basic tests completed!")
    print("=" * 70)
    print("\nOptional: Run a quick training test?")
    print("This will train a model for 3 epochs to verify everything works.")
    quick_training_test()

    print("\n" + "=" * 70)
    print("Example script completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. If all checks passed, run: python train.py")
    print("  2. After training, run: python inference.py --mode evaluate")
    print("  3. Check README.md for detailed documentation")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
