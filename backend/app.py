from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import shap
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("calibrated_model.pkl")

# Background dataset (VERY IMPORTANT for SHAP stability)
background = pd.DataFrame(
    np.zeros((1, 13)),
    columns=[
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"
    ]
)

def predict_fn(X):
    return model.predict_proba(X)[:, 1]

explainer = shap.TreeExplainer(model.estimator)

# =========================
# FEATURES
# =========================
features = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

# =========================
# FEATURE GROUPS (clinical abstraction)
# =========================
feature_groups = {
    "age": "Demographic",
    "sex": "Demographic",
    "cp": "Symptomatology",
    "trestbps": "Hemodynamic",
    "chol": "Metabolic",
    "fbs": "Metabolic",
    "restecg": "Electrical",
    "thalach": "Functional Capacity",
    "exang": "Exercise-induced Ischemia",
    "oldpeak": "Ischemia",
    "slope": "Ischemia",
    "ca": "Coronary Anatomy",
    "thal": "Perfusion"
}

# =========================
# VALIDATION
# =========================
def validate_input(data):
    for f in features:
        if f not in data:
            return f"{f} is required"

    if not (0 <= data["age"] <= 120):
        return "Invalid age"

    return None

# =========================
# RISK CATEGORY
# =========================
def risk_category(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Intermediate Risk"
    else:
        return "High Risk"

# =========================
# CLINICAL REASONING
# =========================
def generate_clinical_reasoning(feature_sorted, prob, risk):
    top = feature_sorted[:3]

    text = f"""
Clinical Cardiovascular Risk Interpretation:

The patient is classified as {risk}, with a predicted probability of {round(prob*100,1)}%.

The dominant drivers of this prediction include:
"""

    for feat, val in top:
        direction = "increasing" if val > 0 else "reducing"
        text += f"- {feat} is a significant factor {direction} overall cardiovascular risk.\n"

    text += """
This pattern suggests underlying:
- Hemodynamic stress
- Possible myocardial ischemia
- Reduced functional cardiovascular reserve

Clinical correlation and further diagnostic evaluation are advised.
"""

    return text

# =========================
# SHAP RECOMMENDATIONS
# =========================
def generate_recommendations(feature_sorted):
    recommendations = []

    for feat, val in feature_sorted:
        if val > 0:
            recommendations.append(f"Consider optimizing {feat}")

    return recommendations[:5]

@app.route("/health")
def health():
    return {"status": "ok"}, 200


# =========================
# PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # VALIDATION
        error = validate_input(data)
        if error:
            return jsonify({"status": "error", "error": error}), 400

        df = pd.DataFrame([data])[features]

        # PREDICTION
        prob = float(model.predict_proba(df)[0][1])
        risk = risk_category(prob)

        # CONFIDENCE
        confidence = abs(prob - 0.5) * 2

        # SHAP
        shap_values = explainer(df)
        values = shap_values.values[0]
        base_value = float(shap_values.base_values[0])

        feature_impact = list(zip(features, values))

        feature_sorted = sorted(
            feature_impact,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanation = [
            {
                "feature": f,
                "impact": float(v),
                "direction": "increases" if v > 0 else "decreases",
                "group": feature_groups.get(f, "Other")
            }
            for f, v in feature_sorted[:5]
        ]

        # RECOMMENDATIONS (SHAP-driven)
        recommendations = generate_recommendations(feature_sorted)

        # CLINICAL TEXT
        clinical_reasoning = generate_clinical_reasoning(
            feature_sorted,
            prob,
            risk
        )

        return jsonify({
            "status": "success",
            "probability": prob,
            "risk_category": risk,
            "confidence": round(confidence, 3),

            "explanation": explanation,

            "shap_values": values.tolist(),
            "base_value": base_value,
            "feature_names": features,

            "recommendations": recommendations,
            "clinical_reasoning": clinical_reasoning,

            "patient_summary": {
                "age": data.get("age"),
                "sex": data.get("sex"),
                "risk_percent": round(prob * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# =========================
# WHAT-IF ENDPOINT
# =========================
@app.route("/what-if", methods=["POST"])
def what_if():
    try:
        data = request.json

        original = data["original"]
        modified = data["modified"]

        df1 = pd.DataFrame([original])[features]
        df2 = pd.DataFrame([modified])[features]

        p1 = float(model.predict_proba(df1)[0][1])
        p2 = float(model.predict_proba(df2)[0][1])

        return jsonify({
            "status": "success",
            "original_risk": p1,
            "new_risk": p2,
            "delta": p2 - p1
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)