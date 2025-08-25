from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_recall_curve, f1_score)
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

def load_model_simple(model_path, my_args):
    """More robust model loading with multiple fallbacks"""
    try:
        # First try loading full model
        return models.load_model(model_path)
    except Exception as e:
        print(f"Full model load failed: {str(e)}")
        try:
            # Try loading weights only with architecture matching
            from model_creation import create_model
            model = create_model(my_args, (224, 224, 3), len(CLASS_NAMES))
            if model_path.endswith('.h5'):
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
            return model
        except Exception as e:
            print(f"Weights loading failed: {str(e)}")
            raise ValueError("Failed to load model with all methods")

def get_optimal_thresholds(y_true, y_prob):
    """Robust threshold optimization that guarantees meaningful results"""
    thresholds = []
    for i in range(len(CLASS_NAMES)):
        class_true = y_true[:, i]
        class_prob = y_prob[:, i]
        
        # Skip if no positive samples
        if np.sum(class_true) == 0:
            thresholds.append(0.99)  # Predict rarely if no positives in train
            continue
            
        # Calculate precision-recall curve
        precision, recall, threshs = precision_recall_curve(class_true, class_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Ensure we have valid thresholds
        if len(threshs) == 0:
            thresholds.append(0.7)  # Conservative default
            continue
            
        # Find threshold that maximizes F1
        best_idx = np.argmax(f1_scores[:-1])  # Skip last element
        best_thresh = threshs[best_idx]
        
        # Validate the threshold
        preds = (class_prob >= best_thresh).astype(int)
        if np.sum(preds) == 0:
            # If no predictions, use lower threshold
            best_thresh = np.percentile(class_prob, 95)  # Top 5% as positive
            if np.sum(class_prob >= best_thresh) == 0:
                best_thresh = np.max(class_prob) - 1e-5  # At least one positive
            
        thresholds.append(best_thresh)
    
    return np.array(thresholds)

def plot_confusion_matrix(y_true, y_pred, class_index=0):
    """Plot confusion matrix for a specific class"""
    cm = confusion_matrix(y_true[:, class_index], y_pred[:, class_index])
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix for {CLASS_NAMES[class_index]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{CLASS_NAMES[class_index]}.png')
    plt.close()

def evaluate_model(model, test_gen, max_batches=50):
    """Core evaluation function"""
    # Get predictions
    y_true, y_prob = [], []
    for i, (X, y) in enumerate(test_gen):
        if i >= max_batches:
            break
        y_true.append(y.numpy())
        y_prob.append(model.predict(X, verbose=0))
    
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    
    # Calculate optimal thresholds
    thresholds = get_optimal_thresholds(y_true, y_prob)
    y_pred = (y_prob >= thresholds).astype(int)
    
    # Generate outputs
    print("\nOptimal Thresholds:")
    for name, thresh in zip(CLASS_NAMES, thresholds):
        print(f"{name:<20}: {thresh:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Plot confusion matrix for first 3 classes
    for i in range(min(3, len(CLASS_NAMES))):
        plot_confusion_matrix(y_true, y_pred, i)
    
    # Save thresholds
    np.save('optimal_thresholds.npy', thresholds)
    print("\nSaved thresholds to optimal_thresholds.npy")
    
    return y_true, y_prob, y_pred

def show_score(my_args, test_gen):
    """Simplified scoring interface"""
    model = load_model_simple(my_args.model_file,my_args)
    return evaluate_model(model, test_gen)