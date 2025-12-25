import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

from src.model import get_model, count_parameters
from src.dataset import load_simple_dataset


class Trainer:
    """Training pipeline for parking space classifier"""

    def __init__(self, model, train_loader, val_loader, device='cuda',
                 learning_rate=0.001, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max',
                                          factor=0.5, patience=5, verbose=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        self.best_val_acc = 0.0
        self.best_model_wts = None

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            # Update progress bar
            current_loss = running_loss / total_samples
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({'loss': f'{current_loss:.4f}',
                            'acc': f'{current_acc:.4f}'})

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        return epoch_loss, epoch_acc.item()

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # Store predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                pbar.set_postfix({'loss': f'{current_loss:.4f}',
                                'acc': f'{current_acc:.4f}'})

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        return epoch_loss, epoch_acc.item(), all_preds, all_labels

    def train(self, num_epochs=50, early_stopping_patience=10):
        """
        Complete training loop

        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 70)

        start_time = time.time()
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()

            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                print(f"  Validation accuracy improved: {self.best_val_acc:.4f} -> {val_acc:.4f}")
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0

                # Save checkpoint
                self.save_checkpoint(epoch, val_acc, 'best_model.pth')
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epochs")

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Save latest checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_acc, f'checkpoint_epoch_{epoch+1}.pth')

        # Training complete
        time_elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)

        # Save final model
        self.save_checkpoint(epoch, self.best_val_acc, 'final_model.pth')

        # Save training history
        self.save_history()

        # Plot training curves
        self.plot_training_curves()

        return self.model

    def save_checkpoint(self, epoch, val_acc, filename):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"\nTraining history saved to: {history_path}")

    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Training curves saved to: {plot_path}")
        plt.close()


def main():
    """Main training function"""
    # Configuration
    config = {
        'data_dir': 'data/processed',
        'model_name': 'resnet18',  # Options: simple_cnn, resnet18, mobilenet, efficientnet_b0
        'image_size': (64, 64),
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'early_stopping_patience': 10,
        'checkpoint_dir': 'checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Parking Space Detection - Training")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = load_simple_dataset(
        config['data_dir'],
        image_size=config['image_size'],
        batch_size=config['batch_size']
    )

    if train_loader is None:
        print("\nERROR: Could not load dataset!")
        print(f"Please ensure dataset is prepared in: {config['data_dir']}")
        print("Run the dataset preparation script first.")
        return

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = get_model(
        model_name=config['model_name'],
        num_classes=2,
        pretrained=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        checkpoint_dir=config['checkpoint_dir']
    )

    # Train model
    trained_model = trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )

    print("\n" + "=" * 70)
    print("Training complete! Model saved to checkpoints directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
