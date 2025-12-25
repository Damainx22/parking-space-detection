"""
Configuration file for parking space detection
Modify these settings to customize your training
"""

import torch


class Config:
    """Training configuration"""

    # ==================== Data Settings ====================
    DATA_DIR = 'data/processed'
    IMAGE_SIZE = (64, 64)  # (height, width)
    BATCH_SIZE = 32

    # ==================== Model Settings ====================
    # Options: 'simple_cnn', 'resnet18', 'mobilenet', 'efficientnet_b0'
    MODEL_NAME = 'resnet18'
    NUM_CLASSES = 2  # Empty (0) and Occupied (1)
    USE_PRETRAINED = True  # Use ImageNet pre-trained weights

    # ==================== Training Settings ====================
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs

    # ==================== Paths ====================
    CHECKPOINT_DIR = 'checkpoints'
    RESULTS_DIR = 'results'

    # ==================== Hardware ====================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0  # Data loading workers (set to 0 on Windows)

    # ==================== Advanced Settings ====================
    # Optimizer
    OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
    WEIGHT_DECAY = 0.0001  # L2 regularization
    MOMENTUM = 0.9  # For SGD only

    # Learning rate scheduler
    LR_SCHEDULER = 'plateau'  # Options: 'plateau', 'step', 'cosine'
    LR_FACTOR = 0.5  # Multiply LR by this when plateau detected
    LR_PATIENCE = 5  # Epochs to wait before reducing LR

    # Data augmentation (applied during training)
    AUGMENTATION = True
    RANDOM_FLIP = True
    RANDOM_ROTATION = 10  # degrees
    COLOR_JITTER = True
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2

    # Normalization (ImageNet statistics)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        config_dict = {}
        for key in dir(cls):
            if key.isupper():
                config_dict[key.lower()] = getattr(cls, key)
        return config_dict

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 70)
        print("Configuration")
        print("=" * 70)

        print("\nData:")
        print(f"  Data directory: {cls.DATA_DIR}")
        print(f"  Image size: {cls.IMAGE_SIZE}")
        print(f"  Batch size: {cls.BATCH_SIZE}")

        print("\nModel:")
        print(f"  Architecture: {cls.MODEL_NAME}")
        print(f"  Pre-trained: {cls.USE_PRETRAINED}")
        print(f"  Number of classes: {cls.NUM_CLASSES}")

        print("\nTraining:")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Early stopping patience: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"  Device: {cls.DEVICE}")

        print("\nPaths:")
        print(f"  Checkpoints: {cls.CHECKPOINT_DIR}")
        print(f"  Results: {cls.RESULTS_DIR}")

        print("=" * 70)


# ==================== Preset Configurations ====================

class QuickTestConfig(Config):
    """Fast configuration for testing the pipeline"""
    NUM_EPOCHS = 3
    BATCH_SIZE = 8
    MODEL_NAME = 'simple_cnn'
    EARLY_STOPPING_PATIENCE = 2


class AccuracyConfig(Config):
    """Configuration optimized for best accuracy"""
    MODEL_NAME = 'resnet18'
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 15
    USE_PRETRAINED = True


class SpeedConfig(Config):
    """Configuration optimized for fast training/inference"""
    MODEL_NAME = 'mobilenet'
    IMAGE_SIZE = (48, 48)
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    USE_PRETRAINED = True


class CPUConfig(Config):
    """Configuration for CPU-only training"""
    DEVICE = 'cpu'
    BATCH_SIZE = 16
    MODEL_NAME = 'simple_cnn'
    NUM_WORKERS = 0


# ==================== Select Configuration ====================
# Change this to use different presets
ACTIVE_CONFIG = Config  # Options: Config, QuickTestConfig, AccuracyConfig, SpeedConfig, CPUConfig


if __name__ == "__main__":
    print("\nAvailable Configurations:")
    print("-" * 70)
    print("1. Config (Default) - Balanced settings")
    print("2. QuickTestConfig - Fast testing (3 epochs)")
    print("3. AccuracyConfig - Best accuracy (100 epochs, ResNet18)")
    print("4. SpeedConfig - Fast training (MobileNet)")
    print("5. CPUConfig - CPU-only training")
    print("-" * 70)
    print("\nCurrent Configuration:")
    ACTIVE_CONFIG.print_config()
