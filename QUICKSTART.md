# Quick Start Guide

Get started with parking space detection in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

## Step 2: Get Sample Dataset (2 min)

### Option A: Download PKLot (Recommended)

1. Download: https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
2. Extract to `data/raw/PKLot/`
3. Process dataset:

```python
from src.dataset import PKLotDatasetExtractor

extractor = PKLotDatasetExtractor('data/raw/PKLot', 'data/processed')
extractor.process_dataset()
```

### Option B: Use Your Own Images

Organize images into:
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

## Step 3: Train Your First Model (5-20 min)

```bash
python train.py
```

That's it! The script will:
- Load your dataset
- Train a ResNet18 model
- Save the best model to `checkpoints/`
- Create training curves

## Step 4: Test Your Model (1 min)

```bash
python inference.py --mode evaluate
```

## Step 5: Predict on New Image

```bash
python inference.py --mode predict --image path/to/your/image.jpg
```

## Quick Configuration Changes

Edit `train.py` to change settings:

```python
# Use a lighter model
config['model_name'] = 'mobilenet'  # Faster training

# Train on CPU
config['device'] = 'cpu'  # If no GPU available

# Quick test run
config['num_epochs'] = 10  # Reduce epochs for testing
```

## Expected Results

After training completes, you should see:
- **Accuracy**: 90-97% (depending on model and dataset)
- **Training time**: 10-25 minutes (with GPU)
- **Model saved**: `checkpoints/best_model.pth`
- **Plots**: `checkpoints/training_curves.png`

## Troubleshooting

### "Could not load dataset"
- Check that images are in `data/processed/train/empty/` and `data/processed/train/occupied/`
- Ensure images are in `.jpg` format

### "CUDA out of memory"
```python
# In train.py, reduce batch size:
config['batch_size'] = 16  # or even 8
```

### "No module named 'src'"
```bash
# Make sure you're in the project root directory
cd ParkingSpaceDetection
python train.py
```

## Next Steps

1. Check `README.md` for detailed documentation
2. Visualize your dataset: `python -c "from utils.visualize import *; visualize_dataset_samples('data/processed')"`
3. Experiment with different models
4. Try hyperparameter tuning

## Common Commands Cheat Sheet

```bash
# Train model
python train.py

# Evaluate on test set
python inference.py --mode evaluate

# Predict single image
python inference.py --mode predict --image image.jpg

# Visualize prediction
python inference.py --mode visualize --image image.jpg

# Use different model
python inference.py --mode evaluate --model mobilenet --checkpoint checkpoints/best_model.pth
```

## Tips for Best Results

1. **Balanced Dataset**: Have roughly equal empty/occupied images
2. **Data Quality**: Clear images, consistent lighting
3. **Model Choice**:
   - ResNet18 = Best accuracy
   - MobileNet = Fastest
   - SimpleCNN = Best for learning
4. **GPU Training**: Much faster than CPU (20x speedup)
5. **Patience**: Let early stopping work (don't interrupt training)

## Getting Help

1. Read error messages carefully
2. Check `README.md` for detailed docs
3. Review training logs in terminal
4. Visualize your data to ensure it's correct

Happy training!
