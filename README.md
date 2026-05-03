# Clinical Cardiovascular Risk Assessment System

## Overview

This project is a full-stack clinical decision support system designed to estimate cardiovascular disease risk using machine learning. It integrates predictive modeling, explainable artificial intelligence, and automated report generation to support clinical interpretation.

The system is intended for educational and research purposes and is not a substitute for professional medical diagnosis.

---

## Key Features

* Structured patient clinical data input
* Machine learning–based cardiovascular risk prediction
* Risk categorization (Low, Intermediate, High)
* Probability estimation with credibility scoring
* Explainable artificial intelligence insights
* Clinical reasoning output
* Automated clinical recommendations
* Downloadable cardiology consultation report (Portable Document Format)
* Mobile-friendly user interface

---

## System Architecture

Frontend:

* Streamlit-based interactive clinical interface
* Visualization of risk metrics and results
* Report generation using ReportLab

Backend:

* Flask-based application programming interface
* Machine learning model inference
* Clinical reasoning and recommendation engine

---

## Project Structure

```
cardio-ai/
│
├── backend/
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── Procfile
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---
## Installation and Local Setup
###Demo Link: https://yusuf-cardio-ai.streamlit.app/

## Installation and Local Setup

### 1. Clone Repository

```
git clone https://github.com/GrumpyCat99/cardio-ai.git
cd cardio-ai
```

---

### 2. Backend Setup

```
cd backend
pip install -r requirements.txt
python app.py
```

---

### 3. Frontend Setup

```
cd ../frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Application Programming Interface

### Endpoint

POST `/predict`

### Input Example

```
{
  "age": 58,
  "sex": 1,
  "cp": 0,
  "trestbps": 140,
  "chol": 260,
  "fbs": 0,
  "restecg": 1,
  "thalach": 120,
  "exang": 1,
  "oldpeak": 2.5,
  "slope": 1,
  "ca": 2,
  "thal": 3
}
```

### Output Example

```
{
  "status": "success",
  "probability": 0.82,
  "confidence": 0.64,
  "risk_category": "High",
  "clinical_reasoning": "...",
  "recommendations": ["..."]
}
```

---

## Deployment

### Backend Deployment

* Platform: Render
* Start command:

```
gunicorn app:app
```

---

### Frontend Deployment

* Platform: Streamlit Community Cloud
* Entry file:

```
frontend/app.py
```

---

## Clinical Interpretation

* **Risk Category**
  Provides a categorical stratification of cardiovascular risk to guide clinical urgency.

* **Probability**
  Represents the predicted likelihood of cardiovascular disease.

* **Credibility**
  Reflects the confidence of the model prediction based on uncertainty estimation.

---

## Limitations

* The model is trained on a limited dataset and may not generalize to all populations
* Not validated for clinical deployment
* Should not be used as a standalone diagnostic tool

---

## Future Improvements

* Model validation with external datasets
* Calibration and performance metrics reporting
* Integration with electronic health record systems
* Authentication and patient data management
* Real-time explainability optimization

---

## Author

Ibrahim Muhammad Yusuf
Medical Doctor | Full Stack Developer | Healthcare Artificial Intelligence

---

## License

This project is intended for educational and portfolio purposes only.
