import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
from sklearn.metrics import roc_curve, confusion_matrix

def save_roc_plot(y_test, y_probs, save_path='../results/roc_curve.png'):
    """Computes and saves the ROC curve plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def save_confusion_matrix_plot(y_test, y_preds, save_path='../results/confusion_matrix.png'):
    """Generates and saves the confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No CAD', 'CAD'], yticklabels=['No CAD', 'CAD'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

def load_single_image(image_path, img_size=(224, 224)):
    """Loads and resizes an eye image, normalized to [0, 1] with batch dims."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    img = cv2.resize(img, img_size)
    img_array = img.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 224, 224, 3)
    return img_batch
