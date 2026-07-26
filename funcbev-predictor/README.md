# 🥤 Functional Beverage Adoption Predictor

A machine learning-powered web application that predicts whether a consumer is likely to adopt functional beverages (e.g., energy drinks, health tonics, probiotic drinks) based on their psychographic and behavioral survey responses.

Built as part of a Master's thesis research project using the **Theory of Planned Behavior (TPB)** framework.

---

## 📋 Project Overview

This tool uses a trained classification model to analyze six key psychological constructs and predict consumer adoption behavior:

| Construct | Description |
|-----------|-------------|
| **HO** | Health Orientation |
| **AT** | Attitude toward Functional Beverages |
| **BR** | Brand Perception |
| **SN** | Subjective Norms |
| **PBC** | Perceived Behavioral Control |
| **INNO** | Innovativeness |

Respondents answer a 5-point Likert scale survey and receive an instant prediction with a confidence score and per-construct breakdown.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **ML Model**: Scikit-learn (trained on survey data)
- **Frontend**: Vanilla HTML/CSS/JavaScript (served as static files)
- **Data Preprocessing**: NumPy, Scikit-learn StandardScaler

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/vinmehta-star/Functional-Beverage-Adoption-Predictor.git
cd Functional-Beverage-Adoption-Predictor

# 2. Create and activate a virtual environment
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# From inside the backend/ directory
python main.py
```

Then open your browser and go to: **http://127.0.0.1:8000**

---

## 📁 Project Structure

```
funcbev-predictor/
├── backend/
│   ├── main.py              # FastAPI backend server
│   ├── model.pkl            # Trained ML classification model
│   ├── scaler.pkl           # Fitted StandardScaler
│   ├── requirements.txt     # Python dependencies
│   ├── test_predict.py      # Unit tests for prediction endpoint
│   └── static/
│       └── index.html       # Frontend UI (survey + results)
├── ML_pro.ipynb             # Jupyter notebook: EDA, model training & evaluation
├── Master Thesis Survey (Responses).csv  # Raw survey data
└── .gitignore
```

---

## 🧪 Running Tests

```bash
# From inside the backend/ directory
python test_predict.py
```

---

## 📊 Model Details

- **Algorithm**: Logistic Regression / Classification (trained in `ML_pro.ipynb`)
- **Features**: 21 survey items across 6 constructs, with reverse-coded items
- **Reverse Coded Items**: AT1, AT3, BR1, BR4, SN4, INNO1, INNO4
- **Preprocessing**: StandardScaler normalization

---

## 📄 License

This project is for academic research purposes.
