# 🍌 Banana Spoilage Detection — CNN Model

A Convolutional Neural Network (CNN) trained to classify bananas as **Fresh** or **Rotten** with ~99.5% accuracy.

---

## 📁 Project Structure

```
kela/
├── FreshBanana/                 # Training images - fresh bananas
├── RottenBanana/                # Training images - rotten bananas
├── banana_spoilage_model.keras  # ✅ Trained model (upload this to GitHub!)
├── train.py                     # Script used to train the model
├── predict.py                   # Script to run predictions on new images
└── requirements.txt             # Python dependencies
```

---

## ✅ Training Results

| Metric              | Value   |
|---------------------|---------|
| Training Accuracy   | ~99.47% |
| Validation Accuracy | ~99.53% |
| Epochs              | 10      |
| Image Size          | 128×128 |
| Dataset Size        | ~7,511 images |

---

## 🚀 How to Run on Another PC (No Training Needed)

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/kela.git
cd kela
```

### Step 2 — Install Python 3.11
> ⚠️ TensorFlow requires **Python 3.9 – 3.12**. Python 3.14+ is NOT supported.
- Download from: https://www.python.org/downloads/release/python-3119/

### Step 3 — Install dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### Step 4 — Run prediction on an image
```bash
py -3.11 predict.py path/to/your/banana_image.jpg
```

**Example output:**
```
--- Analysis for banana.jpg ---
Spoilage level: 87.34% Rotten (12.66% Fresh)
Conclusion: The banana appears to be ROTTEN.
```

---

## 🧠 Model Info

- **Architecture:** CNN (3× Conv2D + MaxPool, Dense 128, Sigmoid output)
- **Classes:** `0 = FreshBanana`, `1 = RottenBanana`
- **Framework:** TensorFlow / Keras
- **Model file:** `banana_spoilage_model.keras` (~25 MB)
