import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    Simple CNN for parking space classification
    Lightweight and fast, good for learning and quick experiments
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for parking space detection
    Uses pre-trained ResNet18 with custom classification head
    Better accuracy than SimpleCNN, still relatively fast
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super(ResNetClassifier, self).__init__()

        # Load pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class MobileNetClassifier(nn.Module):
    """
    MobileNetV2-based classifier
    Extremely lightweight, optimized for mobile/embedded deployment
    Fast inference with good accuracy
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetClassifier, self).__init__()

        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier
    State-of-the-art efficiency and accuracy tradeoff
    """
    def __init__(self, num_classes=2, pretrained=True, model_name='efficientnet_b0'):
        super(EfficientNetClassifier, self).__init__()

        # Load EfficientNet (requires torchvision >= 0.11)
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def get_model(model_name='simple_cnn', num_classes=2, pretrained=True):
    """
    Factory function to get model by name

    Args:
        model_name: One of ['simple_cnn', 'resnet18', 'mobilenet', 'efficientnet_b0']
        num_classes: Number of output classes (default: 2 for occupied/empty)
        pretrained: Whether to use pre-trained weights (for transfer learning models)

    Returns:
        PyTorch model
    """
    model_name = model_name.lower()

    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
        print(f"Created SimpleCNN model")

    elif model_name == 'resnet18':
        model = ResNetClassifier(num_classes=num_classes, pretrained=pretrained)
        print(f"Created ResNet18 classifier (pretrained={pretrained})")

    elif model_name == 'mobilenet':
        model = MobileNetClassifier(num_classes=num_classes, pretrained=pretrained)
        print(f"Created MobileNetV2 classifier (pretrained={pretrained})")

    elif model_name == 'efficientnet_b0':
        model = EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained,
                                       model_name='efficientnet_b0')
        print(f"Created EfficientNet-B0 classifier (pretrained={pretrained})")

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: simple_cnn, resnet18, mobilenet, efficientnet_b0")

    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")

    models_to_test = ['simple_cnn', 'resnet18', 'mobilenet']

    for model_name in models_to_test:
        print(f"\n{model_name.upper()}")
        print("-" * 50)

        # Create model
        model = get_model(model_name, num_classes=2, pretrained=False)

        # Count parameters
        params = count_parameters(model)
        print(f"Trainable parameters: {params:,}")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 64, 64)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        print(f"Model size: {size_mb:.2f} MB")
