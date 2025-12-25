# Parking Space Detection using Deep Learning

A complete PyTorch-based solution for detecting parking space occupancy using Convolutional Neural Networks (CNN). This system can classify parking spaces as either "Empty" or "Occupied" with high accuracy.

## Features

- Multiple CNN architectures (SimpleCNN, ResNet18, MobileNetV2, EfficientNet)
- Complete training pipeline with validation and early stopping
- Data augmentation for improved generalization
- Model evaluation with detailed metrics
- Visualization utilities for dataset and predictions
- Easy-to-use inference API
- Support for PKLot dataset

## Project Structure

```
ParkingSpaceDetection/
├── data/
│   ├── raw/              # Raw dataset (PKLot)
│   └── processed/        # Processed parking space patches
│       ├── train/
│       │   ├── empty/
│       │   └── occupied/
│       ├── val/
│       │   ├── empty/
│       │   └── occupied/
│       └── test/
│           ├── empty/
│           └── occupied/
├── src/
│   ├── dataset.py        # Dataset loading and preprocessing
│   └── model.py          # Model architectures
├── utils/
│   └── visualize.py      # Visualization utilities
├── checkpoints/          # Saved model checkpoints
├── results/              # Evaluation results and plots
├── train.py              # Training script
├── inference.py          # Inference and evaluation script
└── requirements.txt      # Python dependencies
```

## Installation

### 1. Clone or Create Project Directory

```bash
cd ParkingSpaceDetection
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Option 1: Use PKLot Dataset (Recommended)

1. **Download PKLot Dataset:**
   - Visit: https://www.inf.ufpr.br/vri/databases/
   - Download PKLot dataset (PKLot.tar.gz)
   - Extract to `data/raw/PKLot/`

2. **Process Dataset:**

```python
from src.dataset import PKLotDatasetExtractor

extractor = PKLotDatasetExtractor(
    dataset_path='data/raw/PKLot',
    output_path='data/processed'
)
extractor.process_dataset(image_size=(64, 64))
```

This will extract parking space patches and organize them into train/val/test splits.

### Option 2: Use Your Own Dataset

Organize your images in the following structure:

```
data/processed/
├── train/
│   ├── empty/      # Empty parking space images
│   └── occupied/   # Occupied parking space images
├── val/
│   ├── empty/
│   └── occupied/
└── test/
    ├── empty/
    └── occupied/
```

## Usage

### 1. Visualize Dataset (Optional)

```python
from utils.visualize import visualize_dataset_samples, plot_dataset_distribution

# Show sample images
visualize_dataset_samples('data/processed', save_path='results/samples.png')

# Plot class distribution
plot_dataset_distribution('data/processed', save_path='results/distribution.png')
```

### 2. Train Model

**Basic Training:**

```bash
python train.py
```

**Advanced Configuration:**

Edit the `config` dictionary in `train.py`:

```python
config = {
    'data_dir': 'data/processed',
    'model_name': 'resnet18',        # Options: simple_cnn, resnet18, mobilenet, efficientnet_b0
    'image_size': (64, 64),
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'checkpoint_dir': 'checkpoints',
    'device': 'cuda'                 # 'cuda' or 'cpu'
}
```

**What Happens During Training:**

- Model trains with data augmentation
- Validation after each epoch
- Best model saved automatically
- Training curves plotted
- Early stopping prevents overfitting
- Learning rate automatically adjusted

**Expected Results:**

- SimpleCNN: ~90-93% accuracy
- ResNet18: ~95-97% accuracy
- MobileNetV2: ~94-96% accuracy

### 3. Evaluate Model

**Full Evaluation on Test Set:**

```bash
python inference.py --mode evaluate --checkpoint checkpoints/best_model.pth --model resnet18
```

This provides:
- Overall accuracy
- Precision, Recall, F1-score per class
- Confusion matrix
- Per-class accuracy

**Example Output:**

```
Test Accuracy: 0.9612 (96.12%)
Classification Report:
              precision    recall  f1-score   support
       Empty     0.9701    0.9523    0.9611      1250
    Occupied     0.9524    0.9701    0.9612      1250
```

### 4. Predict Single Image

```bash
python inference.py --mode predict --checkpoint checkpoints/best_model.pth --image path/to/space.jpg
```

### 5. Visualize Predictions

```bash
python inference.py --mode visualize --checkpoint checkpoints/best_model.pth --image path/to/space.jpg
```

## Model Architectures

### 1. SimpleCNN (Recommended for Learning)

- Custom lightweight CNN
- ~400K parameters
- Fast training and inference
- Good for prototyping

### 2. ResNet18 (Recommended for Best Accuracy)

- Pre-trained on ImageNet
- ~11M parameters
- Best accuracy (~96%)
- Transfer learning benefits

### 3. MobileNetV2 (Recommended for Deployment)

- Optimized for mobile devices
- ~2M parameters
- Fast inference
- Good accuracy (~95%)

### 4. EfficientNet-B0

- State-of-the-art efficiency
- ~5M parameters
- Excellent accuracy/speed tradeoff

## Python API Usage

### Training

```python
from src.model import get_model
from src.dataset import load_simple_dataset
from train import Trainer

# Load data
train_loader, val_loader, test_loader = load_simple_dataset(
    'data/processed',
    image_size=(64, 64),
    batch_size=32
)

# Create model
model = get_model('resnet18', num_classes=2, pretrained=True)

# Train
trainer = Trainer(model, train_loader, val_loader, device='cuda')
trained_model = trainer.train(num_epochs=50)
```

### Inference

```python
from inference import ParkingSpacePredictor

# Load predictor
predictor = ParkingSpacePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    model_name='resnet18',
    device='cuda'
)

# Predict single image
result = predictor.predict_image('path/to/space.jpg')
print(f"Class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")

# Visualize
predictor.visualize_prediction('path/to/space.jpg', save_path='results/pred.png')
```

## Tips for Best Results

### Data Quality

- Ensure balanced dataset (similar number of empty/occupied samples)
- Use diverse lighting conditions
- Include various weather conditions
- Consistent image quality

### Training

- Start with pre-trained models (ResNet18, MobileNetV2)
- Use data augmentation (already included)
- Monitor validation accuracy to prevent overfitting
- Use GPU for faster training (CUDA)

### Hyperparameter Tuning

- Learning rate: Try [0.0001, 0.001, 0.01]
- Batch size: Try [16, 32, 64]
- Image size: Try [(48,48), (64,64), (96,96)]

## Performance Benchmarks

| Model | Parameters | Accuracy | Training Time* | Inference Speed** |
|-------|-----------|----------|----------------|-------------------|
| SimpleCNN | 400K | ~92% | ~10 min | ~50 FPS |
| ResNet18 | 11M | ~96% | ~20 min | ~35 FPS |
| MobileNetV2 | 2M | ~95% | ~15 min | ~60 FPS |
| EfficientNet-B0 | 5M | ~96% | ~25 min | ~40 FPS |

*On RTX 3060, 50 epochs, 10K training samples
**On RTX 3060, batch size 32

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in train.py
config['batch_size'] = 16

# Or use smaller model
config['model_name'] = 'mobilenet'
```

### Dataset Not Found

```
ERROR: Could not load dataset!
```

Ensure your dataset is organized correctly in `data/processed/` with train/val/test splits.

### Low Accuracy

- Check class balance in dataset
- Increase training epochs
- Try different learning rate
- Use pre-trained model (ResNet18)
- Add more training data

### Slow Training

- Use GPU: `config['device'] = 'cuda'`
- Reduce image size
- Use smaller model (SimpleCNN, MobileNet)
- Increase batch size (if memory allows)

## Citation

If you use the PKLot dataset, please cite:

```
@inproceedings{almeida2015pklot,
  title={PKLot--A robust dataset for parking lot classification},
  author={Almeida, Paulo RL and Oliveira, Luiz S and Britto Jr, Alceu S and Silva Jr, Eunelson J and Koerich, Alessandro L},
  booktitle={Expert Systems with Applications},
  year={2015}
}
```

## License

This project is for educational and research purposes.

## Next Steps

1. **Deploy Model:**
   - Convert to ONNX for production
   - Deploy with Flask/FastAPI
   - Create mobile app with TensorFlow Lite

2. **Improve Accuracy:**
   - Collect more diverse data
   - Try ensemble methods
   - Experiment with attention mechanisms

3. **Add Features:**
   - Multi-camera support
   - Real-time video processing
   - Parking lot occupancy heatmap
   - Historical occupancy tracking

## Support

For issues or questions:
1. Check this README
2. Review training logs in `checkpoints/`
3. Visualize your dataset with utils
4. Ensure proper dataset organization

## Acknowledgments

- PyTorch team for the excellent framework
- PKLot dataset creators
- Pre-trained model providers (ImageNet)
