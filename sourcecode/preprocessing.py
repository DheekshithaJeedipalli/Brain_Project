"""
Brain Hemorrhage Detection using ResNet50
Preprocessing Pipeline

Author: <Your Name>

Description:
This script performs:
1. Dataset Loading
2. Dataset Balancing
3. Train-Validation-Test Split
4. DICOM Image Preprocessing (Hounsfield Unit Conversion, Brain Windowing, Normalization, Resizing)
5. PNG Conversion and Saving
6. Zipping Processed Dataset
"""

import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split

from utils import display_images

# ============================================================
# Dataset Paths
# ============================================================

BASE_PATH = "/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "stage_2_train")
TRAIN_CSV = os.path.join(BASE_PATH, "stage_2_train.csv")

OUTPUT_DIR = "/kaggle/working/rsna_split"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

PROCESSED_DIR = "/kaggle/working/processed_images"
PROCESSED_TRAIN = os.path.join(PROCESSED_DIR, "train")
PROCESSED_VAL = os.path.join(PROCESSED_DIR, "val")
PROCESSED_TEST = os.path.join(PROCESSED_DIR, "test")

# ============================================================
# Hemorrhage Labels
# ============================================================

LABELS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural"
]

# ============================================================
# DICOM Preprocessing Functions
# ============================================================

def preprocess_dicom(dicom_path, target_size=(224, 224), window_center=40, window_width=80):
    """
    Preprocess a CT scan DICOM image:
    1. Read DICOM image.
    2. Convert pixel values to Hounsfield Units (HU).
    3. Apply Brain Windowing.
    4. Normalize pixel values.
    5. Resize image.
    """
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array.astype(np.float32)

    # Convert to Hounsfield Units (HU)
    slope = getattr(dicom, "RescaleSlope", 1.0)
    intercept = getattr(dicom, "RescaleIntercept", 0.0)
    image = image * slope + intercept

    # Brain Windowing
    min_value = window_center - (window_width / 2.0)
    max_value = window_center + (window_width / 2.0)
    image = np.clip(image, min_value, max_value)

    # Normalize Image (0 - 1)
    image = (image - min_value) / (max_value - min_value + 1e-6)

    # Resize Image
    image = cv2.resize(image, target_size)
    return image


def process_dataset(df, output_folder):
    """
    Converts every DICOM image in a dataframe into PNG format after preprocessing.
    """
    processed = 0
    for _, row in df.iterrows():
        dicom_path = row["path"]
        if not os.path.exists(dicom_path):
            continue

        image = preprocess_dicom(dicom_path)
        image_name = row["ImageID"] + ".png"
        save_path = os.path.join(output_folder, image_name)

        # Scale image to 0-255 range and save
        cv2.imwrite(save_path, (image * 255).astype(np.uint8))
        processed += 1

        if processed % 500 == 0:
            print(f"{processed} images processed...")

    print(f"Finished processing {processed} images.")


# ============================================================
# Main Preprocessing Execution
# ============================================================

if __name__ == "__main__":
    # Create Output Directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(PROCESSED_TRAIN, exist_ok=True)
    os.makedirs(PROCESSED_VAL, exist_ok=True)
    os.makedirs(PROCESSED_TEST, exist_ok=True)
    print("Output folders and processed subdirectories created successfully.")

    # Load Dataset
    print("Loading dataset...")
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found. Please verify the dataset path.")
    else:
        df_raw = pd.read_csv(TRAIN_CSV)
        print(f"Total Labels: {len(df_raw)}")

        # Convert Label Format
        df_raw[["ImageID", "Diagnosis"]] = df_raw["ID"].str.rsplit("_", n=1, expand=True)
        df = df_raw.pivot_table(
            index="ImageID",
            columns="Diagnosis",
            values="Label",
            aggfunc="max"
        ).reset_index()
        print(f"Total Images: {len(df)}")

        # Dataset Statistics
        normal_images = (df["any"] == 0).sum()
        hemorrhage_images = (df["any"] == 1).sum()
        print(f"Normal Images      : {normal_images}")
        print(f"Hemorrhage Images  : {hemorrhage_images}")

        # Dataset Balancing
        print("Balancing dataset...")
        normal_df = df[df["any"] == 0]
        hem_df = df[df["any"] == 1]

        # Handle datasets smaller than expected size
        sample_size = min(8000, len(normal_df), len(hem_df))
        if sample_size < 8000:
            print(f"Warning: Insufficient images to sample 8000. Sampling {sample_size} instead.")

        normal_sample = normal_df.sample(sample_size, random_state=42)
        hem_sample = hem_df.sample(sample_size, random_state=42)

        df_final = pd.concat([normal_sample, hem_sample])
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced Dataset Size : {len(df_final)}")

        # Add Image Paths
        df_final["path"] = df_final["ImageID"].apply(
            lambda x: os.path.join(TRAIN_IMAGES_PATH, x + ".dcm")
        )
        print("Image paths generated.")

        # Train Validation Test Split
        print("Creating Train Validation Test Split...")
        train_df, temp_df = train_test_split(df_final, test_size=0.30, random_state=42, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

        print(f"Training Images   : {len(train_df)}")
        print(f"Validation Images : {len(val_df)}")
        print(f"Testing Images    : {len(test_df)}")

        # Save Labels
        train_df[["ImageID"] + LABELS].to_csv("/kaggle/working/train_labels.csv", index=False)
        val_df[["ImageID"] + LABELS].to_csv("/kaggle/working/val_labels.csv", index=False)
        test_df[["ImageID"] + LABELS].to_csv("/kaggle/working/test_labels.csv", index=False)
        print("Label CSV files saved.")

        # Process Training Dataset
        print("\nProcessing Training Images...\n")
        process_dataset(train_df, PROCESSED_TRAIN)

        # Process Validation Dataset
        print("\nProcessing Validation Images...\n")
        process_dataset(val_df, PROCESSED_VAL)

        # Process Test Dataset
        print("\nProcessing Test Images...\n")
        process_dataset(test_df, PROCESSED_TEST)

        # Display Sample Images
        print("\nDisplaying Sample Processed Images...")
        try:
            display_images(PROCESSED_TRAIN, 6)
        except Exception as e:
            print(f"Unable to display images: {e}")

        # Zip Processed Images
        print("\nCreating ZIP files...")
        shutil.make_archive("/kaggle/working/train_images", "zip", PROCESSED_TRAIN)
        shutil.make_archive("/kaggle/working/val_images", "zip", PROCESSED_VAL)
        shutil.make_archive("/kaggle/working/test_images", "zip", PROCESSED_TEST)
        print("ZIP files created successfully.")

        print("\n===================================")
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("===================================")
        print(f"Training Images   : {len(train_df)}")
        print(f"Validation Images : {len(val_df)}")
        print(f"Testing Images    : {len(test_df)}")
        print("\nProcessed images are ready for model training.")
