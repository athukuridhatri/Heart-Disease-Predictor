# 🫀 Heart Disease Risk Predictor

A machine learning web application that predicts the likelihood of heart disease using 13 clinical features from the **UCI Cleveland Heart Disease Dataset**.

**Built by:** Dhatri | B.Tech Computer Science (Data Science) | Malla Reddy University

---

## 📌 Project Overview

This project demonstrates an end-to-end Machine Learning pipeline — from data preprocessing and model training to a live Flask web application with real-time predictions. The application accepts 13 clinical parameters and returns a risk assessment with probability scores, confidence levels, and personalised health recommendations.

---

## ✨ Features

- ✅ **Real Dataset** — UCI Cleveland Heart Disease Dataset (278 clean samples)
- ✅ **Logistic Regression** with StandardScaler preprocessing
- ✅ **Full Evaluation Suite** — Accuracy, Precision, Recall, F1, ROC-AUC
- ✅ **Visual Evaluation Plots** — Confusion Matrix & ROC Curve
- ✅ **Professional Prediction Card** — Risk label, probability %, confidence level, recommendation
- ✅ **Server-side Input Validation** with user-friendly error messages
- ✅ **REST API endpoint** (`/predict`) returning structured JSON
- ✅ **PEP-8 compliant**, well-commented Python code
- ✅ **Medical Disclaimer** for responsible deployment

---

## 🛠️ Technologies Used

| Layer            | Technology                              |
|------------------|-----------------------------------------|
| Language         | Python 3.x                              |
| Web Framework    | Flask                                   |
| ML Library       | scikit-learn (Logistic Regression)      |
| Data Processing  | pandas, numpy                           |
| Visualisation    | matplotlib, seaborn                     |
| Frontend         | HTML5, CSS3, Vanilla JavaScript         |

---

## 📊 Dataset Information

| Property         | Detail                                                    |
|------------------|-----------------------------------------------------------|
| Name             | UCI Cleveland Heart Disease Dataset                       |
| Source           | UCI Machine Learning Repository                           |
| Samples          | 278 (clean, no missing values)                            |
| Features         | 13 clinical attributes                                    |
| Target           | Binary — 0 = No Disease, 1 = Disease                     |
| Class Balance    | ~61% No Disease / ~39% Disease                            |

### Features

| Feature    | Description                                      | Type        |
|------------|--------------------------------------------------|-------------|
| `age`      | Age in years                                     | Continuous  |
| `sex`      | Biological sex (1 = Male, 0 = Female)            | Binary      |
| `cp`       | Chest pain type (0 = Asymptomatic … 3 = Typical) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg)                   | Continuous  |
| `chol`     | Serum cholesterol (mg/dl)                        | Continuous  |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = True)       | Binary      |
| `restecg`  | Resting ECG result (0–2)                         | Categorical |
| `thalach`  | Maximum heart rate achieved                      | Continuous  |
| `exang`    | Exercise-induced angina (1 = Yes)                | Binary      |
| `oldpeak`  | ST depression induced by exercise                | Continuous  |
| `slope`    | Slope of peak exercise ST segment (0–2)          | Categorical |
| `ca`       | Number of major vessels coloured by fluoroscopy  | Ordinal     |
| `thal`     | Thalassemia type (1 = Fixed, 2 = Normal, 3 = Reversible) | Categorical |

---

## 🤖 Model Used

**Logistic Regression** (scikit-learn `LogisticRegression`)

- **Why Logistic Regression?** Simple, interpretable, and fast. Each coefficient directly shows which features increase or decrease the predicted risk, making it easy to explain in interviews or clinical settings.
- **Preprocessing:** `StandardScaler` normalises all features to mean = 0, std = 1 — essential for Logistic Regression since it is sensitive to feature scale.
- **Train / Test Split:** Stratified 80 / 20 split (preserves class ratio).
- **Solver:** `lbfgs` — efficient for small datasets.

---

## 📈 Evaluation Metrics

| Metric     | Score  | Interpretation                                               |
|------------|--------|--------------------------------------------------------------|
| Accuracy   | 91.07% | Overall correct predictions on the test set                  |
| Precision  | 86.96% | Of patients flagged as high-risk, ~87% actually had disease  |
| Recall     | 90.91% | The model correctly detected ~91% of all actual cases        |
| F1 Score   | 88.89% | Harmonic mean of Precision and Recall                        |
| ROC-AUC    | 97.19% | Excellent discrimination between disease and no-disease      |

> **Why Recall matters most here:** A false negative (missing a real case) is far more costly than a false positive (unnecessary follow-up). The model's high Recall (91%) means it rarely misses actual heart disease cases.

Evaluation plots are saved to `static/plots/`:
- `confusion_matrix.png`
- `roc_curve.png`

---

## 📁 Project Architecture

```
heart_disease_project/
│
├── app.py               # Flask backend — routes, validation, prediction logic
├── train_model.py       # Training script — loads data, trains, evaluates, saves model
├── heart.csv            # UCI Cleveland Heart Disease Dataset (278 samples)
├── model.pkl            # Serialised model + scaler + metrics (generated by train_model.py)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
│
├── templates/
│   └── index.html       # Frontend UI — form, result card, disclaimer
│
└── static/
    └── plots/
        ├── confusion_matrix.png   # Generated by train_model.py
        └── roc_curve.png          # Generated by train_model.py
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
cd heart-disease-predictor
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model
```bash
python train_model.py
```
This will:
- Load `heart.csv` (UCI Cleveland dataset)
- Train Logistic Regression with StandardScaler
- Print Accuracy, Precision, Recall, F1, and ROC-AUC
- Save `model.pkl`
- Save `static/plots/confusion_matrix.png` and `roc_curve.png`

### Step 5 — Run the Flask app
```bash
python app.py
```

### Step 6 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🔌 API Reference

### `POST /predict`
Accepts 13 clinical features and returns a prediction.

**Request body (JSON):**
```json
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 140,
  "chol": 270, "fbs": 0, "restecg": 0, "thalach": 130,
  "exang": 1, "oldpeak": 2.5, "slope": 1, "ca": 2, "thal": 3
}
```

**Response (JSON):**
```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "probability_disease": 78.4,
  "probability_no_disease": 21.6,
  "confidence": "High",
  "recommendation": "Consult a cardiologist promptly ...",
  "message": "Heart disease detected",
  "top_features": [...]
}
```

### `GET /model-info`
Returns model metadata and all evaluation metrics.

---

## 🔮 Future Improvements

- [ ] Add cross-validation (k-fold) for more robust evaluation
- [ ] Try ensemble models (Random Forest, XGBoost) and compare results
- [ ] Add SHAP explainability charts for feature contributions
- [ ] Build a Docker container for easy deployment
- [ ] Add unit tests with `pytest`
- [ ] Deploy to Render / Railway / Heroku

---

## ⚠️ Medical Disclaimer

This application is developed for **educational purposes only** and should not be used as a substitute for professional medical diagnosis. The predictions generated are based on a machine learning model trained on a limited dataset and may not reflect individual medical conditions accurately. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.

---

## 📬 Contact

**Dhatri** | athukuridhatri@email.com | [LinkedIn](#) | [GitHub](#)
