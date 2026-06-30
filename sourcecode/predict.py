"""
Brain Hemorrhage Detection using ResNet50
Prediction Script

Author: <Your Name>

Description:
1. Load trained model.
2. Select a random test image.
3. Preprocess the image.
4. Predict hemorrhage presence.
5. Identify specific hemorrhage types.
6. Display confidence scores and summary.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from utils import print_prediction

# ============================================================
# Configuration
# ============================================================

BASE_PATH = "/content/drive/MyDrive/hemorrhage"
MODEL_PATH = os.path.join(BASE_PATH, "resnet_multilabel.h5")
TEST_DIR = os.path.join(BASE_PATH, "test_images")

LABELS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural"
]

HEMORRHAGE_THRESHOLD = 0.30
CLASS_THRESHOLD = 0.20

# ============================================================
# Main Prediction Script
# ============================================================

if __name__ == "__main__":
    # Check if model and test images exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}")
    elif not os.path.exists(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
        print(f"Error: Test directory {TEST_DIR} is empty or does not exist.")
    else:
        # Load Trained Model
        print("Loading trained model...")
        model = load_model(MODEL_PATH)
        print("Model Loaded Successfully.")

        # Select Test Image
        image_name = random.choice(os.listdir(TEST_DIR))
        image_path = os.path.join(TEST_DIR, image_name)
        print(f"\nTesting Image : {image_name}")

        # Load and Preprocess Image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        print("Running prediction...")
        prediction = model.predict(img_array)[0]
        print("Prediction Completed.\n")

        # Detect Hemorrhage presence
        max_prob = np.max(prediction)
        if max_prob < HEMORRHAGE_THRESHOLD:
            print("Status: No Hemorrhage Detected")
        else:
            print("Status: Hemorrhage Detected")

        # Detect Bleed Types
        detected = []
        for i, probability in enumerate(prediction):
            if probability > CLASS_THRESHOLD:
                detected.append((LABELS[i], probability))

        detected = sorted(detected, key=lambda x: x[1], reverse=True)

        if len(detected) == 0:
            print("Bleed Type: Uncertain / Under Threshold")
        else:
            print("\nDetected Hemorrhage Type(s):")
            for label, prob in detected:
                print(f"  - {label:<20} {prob*100:.2f}%")

        # Print confidence scores using utility function
        print_prediction(LABELS, prediction)

        # Print summary
        print("\n==============================")
        print("Prediction Summary")
        print("==============================")
        if len(detected) == 0:
            print("Final Classification: Normal Brain CT")
        else:
            print("Final Classification: Hemorrhage Present")
            print("Detected Classes:")
            for label, _ in detected:
                print(f" - {label}")

        print("\nPrediction Finished Successfully.")
