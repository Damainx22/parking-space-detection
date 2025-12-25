import matplotlib.pyplot as plt
import numpy as np

# Training data from your run
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
train_loss = [0.7600, 0.0647, 0.0468, 0.0405, 0.0361, 0.0330, 0.0299, 0.0271, 0.0246]
val_loss = [0.0799, 0.0463, 0.0398, 0.0389, 0.0317, 0.0294, 0.0269, 0.0238, 0.0265]
train_acc = [71.94, 97.77, 98.40, 98.61, 98.77, 98.87, 98.96, 99.05, 99.13]
val_acc = [97.33, 98.44, 98.65, 98.68, 98.95, 99.02, 99.13, 99.33, 99.22]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
ax1.axvline(x=8, color='green', linestyle='--', alpha=0.7, label='Best Model (Epoch 8)')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# Accuracy plot
ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy', linewidth=2, markersize=8)
ax2.plot(epochs, val_acc, 'r-s', label='Validation Accuracy', linewidth=2, markersize=8)
ax2.axvline(x=8, color='green', linestyle='--', alpha=0.7, label='Best Model (Epoch 8)')
ax2.axhline(y=99.33, color='green', linestyle=':', alpha=0.5, label='Best Val Acc: 99.33%')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)
ax2.set_ylim([70, 100])

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
print("Training curves saved to results/training_curves.png")
