import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm

from src.model import get_model
from src.dataset import load_simple_dataset


class ParkingSpacePredictor:
    """Inference class for parking space detection"""

    def __init__(self, checkpoint_path, model_name='resnet18', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = get_model(model_name, num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from: {checkpoint_path}")
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"Device: {self.device}")

        # Define transforms (same as test transforms)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.class_names = ['Empty', 'Occupied']

    @torch.no_grad()
    def predict_image(self, image_path):
        """
        Predict occupancy for a single parking space image

        Args:
            image_path: Path to parking space image

        Returns:
            dict with prediction, confidence, and class name
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        outputs = self.model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        confidence_score = confidence.item()
        class_name = self.class_names[predicted_class]

        return {
            'prediction': predicted_class,
            'class_name': class_name,
            'confidence': confidence_score,
            'probabilities': probabilities.cpu().numpy()[0]
        }

    @torch.no_grad()
    def predict_batch(self, image_paths):
        """Predict for multiple images"""
        results = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            result = self.predict_image(img_path)
            results.append(result)
        return results

    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with image and confidence"""
        # Get prediction
        result = self.predict_image(image_path)

        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Show image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title(f"Prediction: {result['class_name']}\n"
                         f"Confidence: {result['confidence']:.2%}")

        # Show probability bar chart
        probabilities = result['probabilities']
        axes[1].barh(self.class_names, probabilities, color=['green', 'red'])
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Class Probabilities')

        # Add confidence values
        for i, prob in enumerate(probabilities):
            axes[1].text(prob + 0.02, i, f'{prob:.2%}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()

        plt.close()

        return result


def evaluate_model(checkpoint_path, data_dir, model_name='resnet18', device='cuda'):
    """
    Evaluate model on test set with detailed metrics

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to dataset directory
        model_name: Model architecture name
        device: Device to run evaluation on
    """
    print("=" * 70)
    print("Model Evaluation on Test Set")
    print("=" * 70)

    # Load test data
    _, _, test_loader = load_simple_dataset(data_dir, batch_size=32)

    if test_loader is None:
        print("ERROR: Could not load test set")
        return

    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"\nModel: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Device: {device}\n")

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    print("\n" + "=" * 70)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 70)

    # Classification report
    print("\nClassification Report:")
    print("-" * 70)
    class_names = ['Empty', 'Occupied']
    print(classification_report(all_labels, all_preds,
                                target_names=class_names,
                                digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("-" * 70)
    print(f"{'':>10} {'Empty':>10} {'Occupied':>10}")
    print(f"{'Empty':<10} {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"{'Occupied':<10} {cm[1,0]:>10} {cm[1,1]:>10}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')

    # Save confusion matrix
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    cm_path = results_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_path}")
    plt.close()

    # Calculate per-class accuracy
    empty_correct = cm[0, 0]
    empty_total = cm[0, 0] + cm[0, 1]
    occupied_correct = cm[1, 1]
    occupied_total = cm[1, 0] + cm[1, 1]

    print(f"\nPer-Class Accuracy:")
    print(f"  Empty:    {empty_correct}/{empty_total} = {empty_correct/empty_total:.2%}")
    print(f"  Occupied: {occupied_correct}/{occupied_total} = {occupied_correct/occupied_total:.2%}")

    print("\n" + "=" * 70)

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Parking Space Detection Inference')
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'predict', 'visualize'],
                       help='Inference mode')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['simple_cnn', 'resnet18', 'mobilenet', 'efficientnet_b0'],
                       help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to dataset directory')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')

    args = parser.parse_args()

    if args.mode == 'evaluate':
        # Evaluate on test set
        evaluate_model(args.checkpoint, args.data_dir, args.model, args.device)

    elif args.mode == 'predict':
        if args.image is None:
            print("ERROR: --image argument required for predict mode")
            return

        # Single image prediction
        predictor = ParkingSpacePredictor(args.checkpoint, args.model, args.device)
        result = predictor.predict_image(args.image)

        print(f"\nPrediction for: {args.image}")
        print(f"  Class: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        print(f"    Empty: {result['probabilities'][0]:.2%}")
        print(f"    Occupied: {result['probabilities'][1]:.2%}")

    elif args.mode == 'visualize':
        if args.image is None:
            print("ERROR: --image argument required for visualize mode")
            return

        # Visualize prediction
        predictor = ParkingSpacePredictor(args.checkpoint, args.model, args.device)
        result = predictor.visualize_prediction(
            args.image,
            save_path='results/prediction_visualization.png'
        )


if __name__ == "__main__":
    main()
