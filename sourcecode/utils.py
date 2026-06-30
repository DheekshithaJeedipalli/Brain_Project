"""
Brain Hemorrhage Detection
Utility Functions

Author: <Your Name>
"""

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def load_labels(csv_path):
    """
    Load label CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(csv_path)


def display_images(folder, number_of_images=6):
    """
    Display random processed images.

    Parameters:
        folder (str): Path to the folder containing images.
        number_of_images (int): Number of images to display.
    """
    images = os.listdir(folder)[:number_of_images]
    plt.figure(figsize=(12, 8))
    for i, image_name in enumerate(images):
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 3, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(image_name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_loss(history):
    """
    Plot training and validation loss curves.

    Parameters:
        history: Keras training history object.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], marker="o", label="Training Loss")
    plt.plot(history.history["val_loss"], marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_auc(history):
    """
    Plot training and validation AUC curves.

    Parameters:
        history: Keras training history object.
    """
    if "auc" not in history.history:
        print("AUC not available.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["auc"], marker="o", label="Training AUC")
    plt.plot(history.history["val_auc"], marker="s", label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training vs Validation AUC")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_prediction(labels, prediction):
    """
    Print predicted probabilities for each class label.

    Parameters:
        labels (list): Class labels.
        prediction (list): Predicted probabilities.
    """
    print("\n==============================")
    print("Confidence Scores")
    print("==============================")
    for label, prob in zip(labels, prediction):
        print(f"{label:<25}{prob * 100:.2f}%")
