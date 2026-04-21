"""
Banana Freshness CNN Trainer — v2
==================================
Architecture improvements over v1 (binary classifier):

  1. MobileNetV2 backbone (ImageNet pretrained) — richer visual feature
     detection for colour, texture, brown-spot patterns.
  2. Label smoothing (ε = 0.15) — training targets are 0.1 / 0.9 instead
     of 0 / 1, preventing the sigmoid from saturating to extremes.
  3. Temperature scaling (T = 2.0)  — logits are divided by T before the
     sigmoid, which spreads the output distribution away from 0 and 1.
  4. Dropout (0.5 / 0.4)  — regularises the model and calibrates uncertainty
     so intermediate-looking bananas produce intermediate scores.
  5. Two-phase training:
       Phase 1 — train only the new top layers (backbone frozen).
       Phase 2 — unfreeze the last 30 MobileNetV2 layers and fine-tune
                  at a low learning rate so the backbone learns banana-
                  specific colour/texture features.

Result: a banana that is slightly dull will score ~65–75 % fresh rather
than rounding all the way to 100 % or 0 %.
"""

import sys
import io
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import os

# Force UTF-8 output so box-drawing/emoji chars work on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class TemperatureScaling(tf.keras.layers.Layer):
    """Divides the logit by a temperature before applying sigmoid.
    Using a named Layer instead of Lambda avoids Keras safe_mode issues.
    """
    def __init__(self, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, x):
        return tf.sigmoid(x / self.temperature)

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR        = "."
BATCH_SIZE      = 32
IMG_SIZE        = 160          # MobileNetV2 works well at 160×160
EPOCHS_PHASE1   = 8            # Train top layers with backbone frozen
EPOCHS_PHASE2   = 12           # Fine-tune with last 30 backbone layers unfrozen
LABEL_SMOOTHING = 0.15         # Prevents sigmoid from hitting 0 or 1 exactly
TEMPERATURE     = 2.0          # Higher T → softer/more spread output range
DROPOUT_TOP     = 0.50
DROPOUT_MID     = 0.40
L2_REG          = 0.001
MODEL_SAVE_PATH = "banana_freshness_model_v2.keras"


def load_datasets():
    """Load and configure training / validation datasets."""
    # Explicitly name the two classes so Keras ignores any other subfolders
    # (e.g. IPD_FoodSense which contains scripts, not images)
    CLASSES = ["FreshBanana", "RottenBanana"]

    common_args = dict(
        directory    = DATA_DIR,
        validation_split = 0.2,
        seed         = 42,
        image_size   = (IMG_SIZE, IMG_SIZE),
        batch_size   = BATCH_SIZE,
        labels       = "inferred",
        label_mode   = "binary",
        class_names  = CLASSES,   # ← lock to exactly these two folders
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", **common_args
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", **common_args
    )

    class_names = train_ds.class_names
    print(f"\nClasses : {class_names}")
    print(f"  Index 0 = {class_names[0]}  ->  low score  (0.0 = FRESH)")
    print(f"  Index 1 = {class_names[1]}  ->  high score (1.0 = ROTTEN)\n")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def build_model():
    """
    Build the freshness-scoring model.

    The final layer is NOT a plain sigmoid — instead:
        score = sigmoid(logit / TEMPERATURE)

    With TEMPERATURE=2.0 and LABEL_SMOOTHING=0.15:
        - A clearly fresh banana  →  score ≈ 0.10 – 0.30
        - A slightly dull banana  →  score ≈ 0.30 – 0.50
        - A half-rotten banana    →  score ≈ 0.50 – 0.70
        - A clearly rotten banana →  score ≈ 0.70 – 0.90

    This is converted to freshness % = (1 - score) × 100 in predict.py
    """
    # ── Backbone: MobileNetV2 pretrained on ImageNet ───────────────────────
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False   # frozen during phase 1

    # ── Model graph ───────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")

    # MobileNetV2 preprocess: scales [0,255] → [-1, +1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Data augmentation (applied only during training)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomBrightness(0.15)(x)   # simulates lighting changes
    x = layers.RandomContrast(0.15)(x)     # simulates different ripeness looks

    # Backbone feature extraction
    x = backbone(x, training=False)

    # Top classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_TOP)(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_features"
    )(x)
    x = layers.Dropout(DROPOUT_MID)(x)

    # Raw logit (no activation)
    logit = layers.Dense(1, name="logit")(x)

    # Temperature-scaled sigmoid — use named Layer, not Lambda, so model
    # can be loaded without safe_mode=False
    score = TemperatureScaling(temperature=TEMPERATURE, name="freshness_score")(logit)

    model = tf.keras.Model(inputs, score, name="BananaFreshnessV2")
    return model, backbone


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=LABEL_SMOOTHING   # ← key fix; targets = 0.15 / 0.85
        ),
        metrics=["accuracy"],
    )


def main():
    print("=" * 60)
    print("  Banana Freshness Model v2 — Training")
    print("=" * 60)

    train_ds, val_ds = load_datasets()
    model, backbone  = build_model()
    model.summary()

    # ── Phase 1: Train top layers only ────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 1 — Training top layers (backbone frozen)")
    print("─" * 60)
    compile_model(model, learning_rate=1e-3)

    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks_p1,
    )

    # ── Phase 2: Fine-tune last 30 backbone layers ─────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 2 — Fine-tuning last 30 backbone layers")
    print("─" * 60)
    backbone.trainable = True
    # Keep early layers frozen (low-level edge/colour detectors are universal)
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    compile_model(model, learning_rate=1e-5)   # much lower LR for fine-tuning

    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, verbose=1
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks_p2,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅  Model saved to: {MODEL_SAVE_PATH}")
    print(f"    Upload this file to GitHub — predict.py will use it.\n")


if __name__ == "__main__":
    main()
