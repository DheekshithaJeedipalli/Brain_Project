"""
Brain Hemorrhage Detection using ResNet50
Training Pipeline

Author: <Your Name>

Description:
1. Load processed dataset.
2. Create Image Generators.
3. Build Baseline ResNet50 model.
4. Train Baseline Model (frozen base).
5. Fine-Tune Model (last 50 layers unfrozen).
6. Perform final optimization.
7. Save trained model.
8. Plot Performance Graphs.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

# Import helper functions from local utilities
from utils import load_labels, plot_loss, plot_auc

# ============================================================
# Dataset Paths
# ============================================================

BASE_PATH = "/content/drive/MyDrive/hemorrhage"

TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
VAL_DIR = os.path.join(BASE_PATH, "val_images")

TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "val_labels.csv")

MODEL_PATH = os.path.join(BASE_PATH, "resnet_multilabel.h5")

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
# Main Training Pipeline
# ============================================================

if __name__ == "__main__":
    print("Loading Dataset...")
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        print("Error: Label CSV files not found. Check your paths.")
    else:
        train_df = load_labels(TRAIN_CSV)
        val_df = load_labels(VAL_CSV)

        train_df["filename"] = train_df["ImageID"] + ".png"
        val_df["filename"] = val_df["ImageID"] + ".png"

        print(f"Training Images: {len(train_df)}")
        print(f"Validation Images: {len(val_df)}")

        # ============================================================
        # Data Augmentation & Generators
        # ============================================================

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            zoom_range=0.15,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=TRAIN_DIR,
            x_col="filename",
            y_col=LABELS,
            target_size=(224, 224),
            batch_size=16,
            class_mode="raw",
            shuffle=True
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=VAL_DIR,
            x_col="filename",
            y_col=LABELS,
            target_size=(224, 224),
            batch_size=16,
            class_mode="raw",
            shuffle=False
        )

        print("Image Generators Created Successfully")

        # ============================================================
        # Build Baseline ResNet50 Model
        # ============================================================

        print("Building Baseline Model...")
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Freeze all convolutional base layers
        for layer in base_model.layers:
            layer.trainable = False

        # Classification Head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(5, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=outputs)
        print(model.summary())

        # Compile Baseline Model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print("Baseline Model Compiled")

        # Train Baseline Model
        print("\nTraining Baseline Model...\n")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=10
        )
        print("Baseline Training Completed")

        # ============================================================
        # Fine-Tuning ResNet50
        # ============================================================

        print("\nStarting Fine-Tuning...\n")

        # Unfreeze the last 50 layers of ResNet50
        for layer in base_model.layers[-50:]:
            layer.trainable = True

        print("Last 50 layers are now trainable.")

        # Compile Model for Fine-Tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=[AUC(name="auc")]
        )
        print("Fine-Tuning Model Compiled.")

        # Early Stopping Callback
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        )

        # Fine-Tune Model
        history_finetune = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=10,
            callbacks=[early_stop]
        )
        print("Fine-Tuning Completed Successfully.")

        # ============================================================
        # Final Training Stage
        # ============================================================

        print("\nPerforming Final Optimization...\n")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            loss="binary_crossentropy",
            metrics=[AUC(name="auc")]
        )

        history_final = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=3
        )
        print("Final Training Completed.")

        # ============================================================
        # Save Trained Model
        # ============================================================

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        print(f"\nModel Saved Successfully at:\n{MODEL_PATH}")

        # ============================================================
        # Plot Performance Curves
        # ============================================================

        print("\nPlotting Training Loss...")
        plot_loss(history_finetune)

        print("\nPlotting Training AUC...")
        plot_auc(history_finetune)

        print("\n==============================")
        print("TRAINING COMPLETED")
        print("==============================")
        print("Baseline Epochs   : 10")
        print("Fine-Tuning Epochs: 10")
        print("Final Epochs      : 3")
        print(f"Total Training Epochs: 23")
        print("\nModel is ready for prediction.")
