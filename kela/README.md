# 🍌 Banana Freshness Analyser — CNN Model

A Convolutional Neural Network (CNN) that **compares image features** against learned
patterns of fresh and rotten bananas to determine:

- 🟢 **Freshness %** — how close the banana is to perfectly fresh
- 🔴 **Spoilage %**  — how close the banana is to fully rotten
- 📅 **Days remaining** before the banana goes bad (assuming 7 days shelf life from 100 % fresh)

---

## 📁 Project Structure

```
kela/
├── FreshBanana/                 # Training images – fresh bananas  (3 473 images)
├── RottenBanana/                # Training images – rotten bananas (4 038 images)
├── banana_spoilage_model.keras  # ✅ Trained model (~25 MB) — push this to GitHub
├── train.py                     # Script used to train the model
├── predict.py                   # Freshness analyser with shelf-life estimation
└── requirements.txt             # Python dependencies
```

---

## ✅ Training Results

| Metric              | Value     |
|---------------------|-----------|
| Training Accuracy   | **99.47%**|
| Validation Accuracy | **99.53%**|
| Epochs              | 10        |
| Image Size          | 128 × 128 |
| Dataset             | 7 511 images |

---

## 🧠 How It Works

The CNN's final layer uses a **sigmoid activation** that outputs a continuous score between 0 and 1:

```
  0.0  ──────►  ALL features match a FRESH banana  (100 % fresh)
  0.5  ──────►  50 % fresh features, 50 % rotten features
  1.0  ──────►  ALL features match a ROTTEN banana (100 % rotten)
```

This score is converted into human-readable metrics:

```python
freshness_pct = (1 - raw_score) * 100       # e.g. 72.4 %
spoilage_pct  = raw_score       * 100        # e.g. 27.6 %
days_left     = (freshness_pct / 100) * 7   # e.g. 5.1 days
```

### Ripeness Stages

| Freshness %  | Stage    | Advice                          |
|-------------|----------|---------------------------------|
| 90 – 100 %  | 🟢 PERFECT  | Eat anytime — peak quality     |
| 70 – 89 %   | 🟡 GOOD     | Still very fresh, no rush      |
| 50 – 69 %   | 🟠 AGING    | Eat soon — quality declining   |
| 25 – 49 %   | 🔴 SPOILING | Best consumed today!           |
|  0 – 24 %   | ⚫ ROTTEN   | No longer safe to eat          |

---

## 🚀 Running on Another PC (No Retraining Needed)

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/kela.git
cd kela
```

### Step 2 — Install Python 3.11
> ⚠️ TensorFlow requires **Python 3.9 – 3.12**. Python 3.13+ is NOT supported yet.
- Download from: https://www.python.org/downloads/release/python-3119/

### Step 3 — Install dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### Step 4 — Run the freshness analyser
```bash
py -3.11 predict.py path/to/your/banana_image.jpg
```

### Example Output

```
  🍌  BANANA FRESHNESS ANALYSIS
  ────────────────────────────────────────────────────
  Image : banana.jpg
  ────────────────────────────────────────────────────

  Freshness   [████████████████████████░░░░░░░░░░░░░░░░] 61.3%
  Spoilage    [████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 38.7%

  ────────────────────────────────────────────────────
  Stage       : 🟠 AGING
  Advice      : Eat soon — quality declining
  ────────────────────────────────────────────────────

  ⏳  Estimated shelf life remaining:
      4.3 days  (103 hours)

      (Based on: 100% fresh banana lasts 7 days)

  📅  Consume before:  Day 5 from now

  Timeline:
      Day 1  🟢  SAFE
      Day 2  🟢  SAFE
      Day 3  🟡  SAFE
      Day 4  🟡  SAFE       ◀ TODAY
      Day 5  🔴  SAFE
      Day 6  🔴  ⚠ ROTTEN
      Day 7  🔴  ⚠ ROTTEN
  ────────────────────────────────────────────────────
```
