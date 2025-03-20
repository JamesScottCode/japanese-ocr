import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import collections
import random

def display_one_random_image_per_class(mapped, kana):
    plt.close('all') 
    num_classes = len(mapped)
    grid_cols = 5
    grid_rows = math.ceil(num_classes / grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
    axes = axes.flatten()
    for idx, (key, value) in enumerate(mapped.items()):
        if len(value) == 0:
            #  blank spot
            axes[idx].axis('off')
            continue
        sample_img = random.choice(value)
        image = sample_img.squeeze()
        
        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(f"Class: {key} {kana[int(key)]}", fontsize=14)
        axes[idx].axis('off')
    
    # hide any completely unused axes (beyond number of classes)
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_average_images(labels, data, kana):
    plt.close('all')
    labels_int = [np.argmax(label) for label in labels]

    total_classes = max(labels_int) + 1

    class_images = {}
    for idx, label in enumerate(labels_int):
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(data[idx])
    
    avg_images = {}
    for cls in range(total_classes):
        if cls in class_images and len(class_images[cls]) > 0:
            imgs_array = np.array(class_images[cls])
            avg_images[cls] = np.mean(imgs_array, axis=0)

    grid_cols = 5
    grid_rows = math.ceil(total_classes / grid_cols)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
    axes = axes.flatten()
    
    # for each class, plot the average image if available; else leave the cell blank.
    for cls in range(total_classes):
        if cls in avg_images:
            axes[cls].imshow(avg_images[cls].squeeze(), cmap='gray')
            axes[cls].set_title(f"Avg Class {cls}", fontsize=14)
            axes[cls].set_title(f"Class: {cls} {kana[int(cls)]}", fontsize=14)
        else:
            # No image available; turn off the axis for a blank spot.
            axes[cls].axis('off')
        axes[cls].axis('off')
    
    # hide any extra axes beyond the total number of classes.
    for i in range(total_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_pixel_distribution(data):
    # flatten all images into a single array for histogramming
    pixel_values = data.ravel()
    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values, bins=30, color='gray', edgecolor='black')
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Normalized Intensity")
    plt.ylabel("Frequency")
    plt.show()

def plot_class_distribution(labels):
    # convert one-hot labels back to class indices
    labels = [np.argmax(label) for label in labels]
    counter = collections.Counter(labels)
    classes = list(counter.keys())
    counts = list(counter.values())
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

# PCA and t-SNE
def plot_dimensionality_reduction(data, labels):
    X = data.reshape((data.shape[0], -1))
    labels = np.array(labels)
    y = np.argmax(labels, axis=1)
    
    cmap = plt.cm.get_cmap('nipy_spectral', 46)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, alpha=0.7)
    plt.title("PCA of Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    cbar = plt.colorbar(scatter, ticks=range(46))
    cbar.set_label("")  
    cbar.ax.set_yticklabels([])
    plt.show()
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, alpha=0.7)
    plt.title("t-SNE of Dataset")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    cbar = plt.colorbar(scatter, ticks=range(46))
    cbar.set_label("")
    cbar.ax.set_yticklabels([])
    plt.show()
