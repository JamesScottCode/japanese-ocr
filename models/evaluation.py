import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import random

from transformations.img_transformations import load_and_preprocess_image, resize_to_target, enhance_character_image_adaptive


def plot_training_history(history):
    # acc plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test, class_names=None):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names)) if class_names is not None else np.arange(cm.shape[0])
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    
    # add counts in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def test_single_sample(model, X, y, class_names, mapped):
    idx = np.random.randint(0, len(X))
    sample = X[idx]
    
    true_label_new = np.argmax(y[idx])

    pred_prob = model.predict(np.expand_dims(sample, axis=0))
    pred_label_new = np.argmax(pred_prob)

    plt.imshow(sample.squeeze(), cmap='gray')
    plt.title(f"True: {class_names[true_label_new]}  |  Pred: {class_names[pred_label_new]}")
    plt.axis('off')
    plt.show()

def evaluate_single_image(model, filepath, class_names, show_graphs=True):
    img = load_and_preprocess_image(filepath)
    print(f"DEBUG: Original image - mean: {img.mean():.3f}, std: {img.std():.3f}")
    if show_graphs:
        pred_prob = model.predict(np.expand_dims(img, axis=0))
        pred_idx = np.argmax(pred_prob)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Original Predicted: {class_names[pred_idx]}")
        plt.axis('off')
        plt.show()
        # print("Original prediction probabilities:", pred_prob)
    
    enhanced_img = enhance_character_image_adaptive(img)
    # print(f"DEBUG: Enhanced image (before resize) - mean: {enhanced_img.mean():.3f}, std: {enhanced_img.std():.3f}")
    
    enhanced_img_resized = resize_to_target(enhanced_img, target_size=(72,76))
    # print(f"DEBUG: Enhanced image (after resize) - mean: {enhanced_img_resized.mean():.3f}, std: {enhanced_img_resized.std():.3f}")
    
    pred_prob_enh = model.predict(np.expand_dims(enhanced_img_resized, axis=0))
    pred_idx_enh = np.argmax(pred_prob_enh)
    if show_graphs:
        plt.imshow(enhanced_img_resized.squeeze(), cmap='gray')
        plt.title(f"Enhanced Predicted: {class_names[pred_idx_enh]}")
        plt.axis('off')
        plt.show()
    return class_names[pred_idx_enh]
    # print("Enhanced prediction probabilities:", pred_prob_enh)