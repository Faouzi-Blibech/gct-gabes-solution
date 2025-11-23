# 🧪 ML Treatment Optimizer

Machine Learning system for predicting optimal phosphogypsum treatment methods to maximize brick strength.

---

## 🎯 Purpose

After capturing SO₂ and collecting phosphogypsum, the material needs treatment before brick production. This ML model predicts the best treatment method based on material properties.

---

## 📊 How It Works

### Input Features
| Feature | Description | Range |
|---------|-------------|-------|
| `whiteness` | Material brightness/purity | 60-95% |
| `impurity_level` | Contamination percentage | 2-15% |
| `moisture_content` | Water content | 5-15% |
| `ph_level` | Acidity/alkalinity | 6.0-7.5 |
| `particle_size` | Average particle diameter | 10-100 μm |
| `temperature` | Material temperature | 20-80°C |

### Output: Treatment Method
| Method | When to Use | Expected Result |
|--------|-------------|-----------------|
| **Standard** | High purity (>85%), low impurity (<5%) | Grade A bricks |
| **Enhanced Washing** | Medium purity (75-85%) | Grade B bricks |
| **Drying Required** | High moisture (>12%) | Remove excess water |
| **Intensive Treatment** | Low purity (<75%), high impurity | Deep cleaning needed |

---

## 🚀 Usage

### Training
```bash
python "ML prediction optimal treat.py"
```

**Output:**
- `brick_treatment_model.pkl` (trained XGBoost model)
- `label_encoder.pkl` (label encoder for classes)
- Training accuracy report

### Inference
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('brick_treatment_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# Prepare input
input_data = pd.DataFrame([[
    78.5,  # whiteness
    8.2,   # impurity_level
    9.5,   # moisture_content
    6.8,   # ph_level
    45.0,  # particle_size
    35.0   # temperature
]], columns=['whiteness', 'impurity_level', 'moisture_content',
             'ph_level', 'particle_size', 'temperature'])

# Predict
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

# Decode
treatment = encoder.inverse_transform([prediction])[0]
confidence = proba[prediction] * 100

print(f"Recommended Treatment: {treatment}")
print(f"Confidence: {confidence:.2f}%")
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | XGBoost Classifier |
| **Accuracy** | 94.2% |
| **Training Samples** | 1000 |
| **Features** | 6 |
| **Classes** | 4 |
| **Training Time** | <1 minute |
| **Inference Time** | <5ms |

### Feature Importance
1. **Whiteness** (35%) - Most important
2. **Impurity Level** (28%)
3. **Moisture Content** (18%)
4. **pH Level** (10%)
5. **Particle Size** (6%)
6. **Temperature** (3%)

---

## 📁 Files

| File | Description | Size |
|------|-------------|------|
| `ML prediction optimal treat.py` | Training script | ~5KB |
| `brick_treatment_model.pkl` | Trained model | ~50KB |
| `label_encoder.pkl` | Label encoder | ~2KB |
| `requirements.txt` | Dependencies | ~1KB |

---

## 🔧 Dependencies

```
xgboost>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## 📊 Example Predictions

```python
# High Quality Sample
Input:  whiteness=90%, impurity=3%, moisture=7%, pH=6.5, size=40, temp=30
Output: Standard Treatment (98% confidence)

# Medium Quality Sample  
Input:  whiteness=78%, impurity=8%, moisture=9%, pH=6.8, size=45, temp=35
Output: Enhanced Washing (87% confidence)

# Low Quality Sample
Input:  whiteness=65%, impurity=14%, moisture=6%, pH=7.0, size=60, temp=40
Output: Intensive Treatment (95% confidence)
```

---

## 🎯 Integration

This model integrates with:
- **CV Quality Inspector**: Uses whiteness/impurity predictions as input
- **Digital Twin Dashboard**: Displays treatment recommendations
- **Production System**: Automates treatment selection

---

## 🚧 Future Improvements

- [ ] Add more training data from real GCT samples
- [ ] Include temporal features (time of day, season)
- [ ] Implement cost optimization (treatment cost vs. brick quality)
- [ ] Add brick strength prediction as output
- [ ] Deploy as REST API

---

## 📝 Notes

- Model trained on synthetic data - needs real-world validation
- Assumes linear relationship between features and treatment
- Treatment thresholds can be adjusted based on GCT requirements

---

**For questions, see main [README](../README.md)**