"""
Banana Freshness Analyser
=========================
Uses the trained CNN model to compare the image against learned features of
FRESH and ROTTEN bananas, then outputs:
  - Freshness %   (how close it is to a perfectly fresh banana)
  - Spoilage  %   (how close it is to a fully rotten banana)
  - Ripeness Stage (Perfect / Good / Aging / Spoiling / Rotten)
  - Days remaining before the banana goes bad
    (Assumption: a 100 % fresh banana lasts 7 days)

Usage:
    py -3.11 predict.py <path_to_image>
    py -3.11 predict.py banana.jpg
    py -3.11 predict.py "C:/photos/my_banana.png"
"""

import sys
import io
import math
import tensorflow as tf
import numpy as np

# Force UTF-8 output on Windows so emoji/box-drawing characters render correctly
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Custom layer (must match train.py exactly for model to load) ──────────────
class TemperatureScaling(tf.keras.layers.Layer):
    """Temperature-scaled sigmoid — prevents output saturating to 0 or 1."""
    def __init__(self, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, x):
        return tf.sigmoid(x / self.temperature)

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


# ── Configuration ────────────────────────────────────────────────────────────
IMG_HEIGHT    = 160        # Must match training image size (v2 model uses 160)
IMG_WIDTH     = 160
MAX_DAYS      = 7          # Days a 100 % fresh banana lasts before fully rotten
BAR_WIDTH     = 40         # Width of the ASCII progress bar
MODEL_PATH    = "banana_freshness_model_v2.keras"

# ── Ripeness stage thresholds (freshness %) ────────────────────────────────
STAGES = [
    (90, "🟢 PERFECT",  "Eat anytime — peak quality"),
    (70, "🟡 GOOD",     "Still very fresh, no rush"),
    (50, "🟠 AGING",    "Eat soon — quality declining"),
    (25, "🔴 SPOILING", "Best consumed today!"),
    ( 0, "⚫ ROTTEN",   "No longer safe to eat"),
]


def make_bar(freshness_pct: float, width: int = BAR_WIDTH) -> str:
    """Return a colour-coded ASCII freshness bar."""
    filled = round(freshness_pct / 100 * width)
    empty  = width - filled

    if freshness_pct >= 70:
        block = "█"
        color_start, color_end = "\033[92m", "\033[0m"   # bright green
    elif freshness_pct >= 40:
        block = "█"
        color_start, color_end = "\033[93m", "\033[0m"   # yellow
    else:
        block = "█"
        color_start, color_end = "\033[91m", "\033[0m"   # red

    bar = f"{color_start}{block * filled}{color_end}{'░' * empty}"
    return f"[{bar}] {freshness_pct:.1f}%"


def get_stage(freshness_pct: float):
    """Return (label, advice) for the given freshness percentage."""
    for threshold, label, advice in STAGES:
        if freshness_pct >= threshold:
            return label, advice
    return STAGES[-1][1], STAGES[-1][2]


def days_remaining(freshness_pct: float) -> float:
    """
    Linear model:
        100 % fresh  →  MAX_DAYS days left
          0 % fresh  →  0 days left
    Formula: days = (freshness_pct / 100) × MAX_DAYS
    """
    return (freshness_pct / 100.0) * MAX_DAYS


def predict_freshness(img_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Load the trained model and run inference on the given image.

    The CNN's sigmoid output is a continuous score in [0, 1]:
        0.0  →  the image has ALL traits of a FRESH banana
        1.0  →  the image has ALL traits of a ROTTEN banana
        0.4  →  40 % rotten features, 60 % fresh features

    Returns a dict with all computed metrics.
    """
    # ── Load model ────────────────────────────────────────────────────────
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"TemperatureScaling": TemperatureScaling}
        )
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        print("  Make sure 'banana_freshness_model_v2.keras' is in the same folder.")
        sys.exit(1)

    # ── Load & preprocess image ───────────────────────────────────────────
    try:
        img = tf.keras.utils.load_img(
            img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
    except Exception as e:
        print(f"\n[ERROR] Could not load image '{img_path}': {e}")
        sys.exit(1)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)          # shape: (1, 128, 128, 3)

    # ── Inference ─────────────────────────────────────────────────────────
    # The model outputs ONE sigmoid value per image (binary classification).
    # Because it trained on matching visual features of fresh vs rotten bananas,
    # intermediate values represent how close the image is to each extreme.
    raw_score   = float(model.predict(img_array, verbose=0)[0][0])

    spoilage_pct  = raw_score * 100.0
    freshness_pct = (1.0 - raw_score) * 100.0
    days_left     = days_remaining(freshness_pct)
    stage_label, stage_advice = get_stage(freshness_pct)

    return {
        "img_path":     img_path,
        "raw_score":    raw_score,
        "freshness_pct": freshness_pct,
        "spoilage_pct":  spoilage_pct,
        "days_left":     days_left,
        "stage_label":   stage_label,
        "stage_advice":  stage_advice,
    }


def print_report(result: dict):
    """Pretty-print the analysis report to the terminal."""
    SEP = "─" * 52

    print()
    print(f"  🍌  BANANA FRESHNESS ANALYSIS")
    print(f"  {SEP}")
    print(f"  Image : {result['img_path']}")
    print(f"  {SEP}")

    # Freshness bar
    bar = make_bar(result["freshness_pct"])
    print(f"\n  Freshness   {bar}")

    # Spoilage bar (inverted)
    spoilage_bar = make_bar(100 - result["spoilage_pct"])
    # We want a spoilage bar that fills from the left for spoilage
    # So we show a red bar for spoilage_pct
    sp = result["spoilage_pct"]
    filled = round(sp / 100 * BAR_WIDTH)
    empty = BAR_WIDTH - filled
    if sp <= 30:
        color_start, color_end = "\033[92m", "\033[0m"
    elif sp <= 60:
        color_start, color_end = "\033[93m", "\033[0m"
    else:
        color_start, color_end = "\033[91m", "\033[0m"
    sp_bar = f"[{color_start}{'█' * filled}{color_end}{'░' * empty}] {sp:.1f}%"
    print(f"  Spoilage    {sp_bar}")

    print()
    print(f"  {SEP}")
    print(f"  Stage       : {result['stage_label']}")
    print(f"  Advice      : {result['stage_advice']}")
    print(f"  {SEP}")

    # Shelf-life calculation
    days = result["days_left"]
    hours = days * 24

    if days >= 1:
        print(f"\n  ⏳  Estimated shelf life remaining:")
        print(f"      {days:.1f} days  ({hours:.0f} hours)")
        print(f"\n      (Based on: 100% fresh banana lasts {MAX_DAYS} days)")
        # Show a simple day-by-day timeline
        print()
        print(f"  📅  Consume before:  Day {math.ceil(days)} from now")
        print()
        print("  Timeline:")
        for d in range(1, MAX_DAYS + 1):
            pct_at_day_d = ((MAX_DAYS - d) / MAX_DAYS) * 100
            current_remaining = days
            marker = " ◀ TODAY" if abs(d - math.ceil(days)) == 0 else ""
            gone  = d > days
            icon  = "🟢" if pct_at_day_d >= 70 else ("🟡" if pct_at_day_d >= 40 else "🔴")
            state = "SAFE      " if not gone else "⚠ ROTTEN  "
            print(f"      Day {d}  {icon}  {state} {marker}")
    else:
        print(f"\n  ⚠️   Shelf life remaining : < 1 day ({hours:.0f} hours)")
        print("       Consume immediately or discard.")

    print(f"\n  {SEP}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:  py -3.11 predict.py <path_to_banana_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    result   = predict_freshness(img_path)
    print_report(result)
