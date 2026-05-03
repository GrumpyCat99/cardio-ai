import streamlit as st
import requests
import shap
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import datetime

st.set_page_config(page_title="Clinical Cardiovascular Artificial Intelligence", layout="wide")

st.title("🫀 Clinical Cardiovascular Risk Assessment System")
st.caption("Artificial intelligence assisted clinical decision support with explainability")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This system estimates cardiovascular risk using machine learning "
    "and provides explainable insights to support clinical decision making.\n\n"
    "This AI Project is created by Ibrahim Muhammad Yusuf \n\n"
    "@2026 All Rights Reserved"
)

st.sidebar.subheader("📘 Clinical Definitions")
st.sidebar.markdown("""
- Chest Pain Type: Classification of angina symptoms  
- Fasting Blood Sugar: Indicator of glucose metabolism  
- Resting Electrocardiography: Electrical activity of heart at rest  
- Maximum Heart Rate: Functional cardiovascular capacity  
- ST Segment Depression: Marker of myocardial ischemia  
- Slope of ST Segment: Response of heart during exercise  
- Coronary Vessels: Number of major vessels with obstruction  
- Thallium Test: Myocardial perfusion imaging result  
""")

st.sidebar.subheader("📊 Result Interpretation")
st.sidebar.markdown("""
- Risk Category: A categorical interpretation of cardiovascular risk (low, intermediate, high). It helps guide urgency and clinical management decisions.

- Probability: The numerical likelihood of cardiovascular disease predicted by the model. It provides quantitative risk estimation.

- Credibility: The confidence level of the prediction, derived from the distance from uncertainty. Higher credibility indicates more reliable predictions.
""")

st.sidebar.subheader("⚠️ Warning")
st.sidebar.markdown("""
This AI model is still under development. Due to limited dataset size, it is not yet suitable for real-world clinical use.
""")

# =========================
# MAPPINGS
# =========================
sex_map = {"Female": 0, "Male": 1}

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non anginal Pain": 2,
    "Asymptomatic": 3
}

fbs_map = {"Normal (120 milligrams per deciliter or less)": 0, "Elevated (more than 120 milligrams per deciliter)": 1}

restecg_map = {
    "Normal": 0,
    "ST segment abnormality": 1,
    "Left ventricular hypertrophy": 2
}

exang_map = {"No": 0, "Yes": 1}

slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

thal_map = {
    "Normal Perfusion": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

ca_map = {
    "0 vessels (no obstruction)": 0,
    "1 vessel (mild disease)": 1,
    "2 vessels (moderate disease)": 2,
    "3 vessels (severe disease)": 3
}

# =========================
# INPUT UI
# =========================
st.header("📋 Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧍 Demographics")
    age = st.number_input("Age (years)", 1, 120, 58)
    sex_label = st.selectbox("Biological Sex", list(sex_map.keys()))

    st.subheader("💢 Symptoms")
    cp_label = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    exang_label = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))

with col2:
    st.subheader("🫀 Vital Signs")
    trestbps = st.number_input("Resting Blood Pressure (millimeters of mercury)", 80, 250, 140)
    thalach = st.number_input("Maximum Heart Rate (beats per minute)", 60, 220, 120)

    st.subheader("🧪 Laboratory and Electrocardiography")
    chol = st.number_input("Cholesterol (milligrams per deciliter)", 100, 600, 260)
    fbs_label = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()))
    restecg_label = st.selectbox("Resting Electrocardiography", list(restecg_map.keys()))

st.subheader("📉 Advanced Cardiac Indicators")

st_depression_label = st.selectbox(
    "ST Segment Depression Severity",
    ["No depression","Mild depression","Moderate depression","Severe depression"]
)

st_depression_map = {
    "No depression": 0.0,
    "Mild depression": 1.0,
    "Moderate depression": 2.5,
    "Severe depression": 4.0
}

oldpeak = st_depression_map[st_depression_label]

slope_label = st.selectbox("ST Segment Slope", list(slope_map.keys()))
ca_label = st.selectbox("Coronary Vessel Status", list(ca_map.keys()))
thal_label = st.selectbox("Thallium Test Result", list(thal_map.keys()))

# =========================
# DATA
# =========================
data = {
    "age": age,
    "sex": sex_map[sex_label],
    "cp": cp_map[cp_label],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_map[fbs_label],
    "restecg": restecg_map[restecg_label],
    "thalach": thalach,
    "exang": exang_map[exang_label],
    "oldpeak": oldpeak,
    "slope": slope_map[slope_label],
    "ca": ca_map[ca_label],
    "thal": thal_map[thal_label]
}

# =========================
# SMART RECOMMENDATIONS
# =========================
def improve_recommendations(recs):
    refined = []
    for r in recs:
        if "age" in r.lower():
            continue
        if "chol" in r.lower():
            refined.append("Reduce cholesterol through diet (low saturated fat), exercise, and statin therapy if indicated")
        elif "trestbps" in r.lower():
            refined.append("Control blood pressure with lifestyle modification and antihypertensive therapy")
        elif "thalach" in r.lower():
            refined.append("Improve cardiovascular fitness through supervised exercise training")
        elif "oldpeak" in r.lower():
            refined.append("Evaluate and manage myocardial ischemia with further cardiac testing")
        else:
            refined.append(f"Further clinical evaluation recommended for {r}")
    return refined

# =========================
# PDF GENERATOR
# =========================
def generate_pdf(result, data):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    small_style = styles["Normal"].clone('small')
    small_style.fontSize = 8

    content = []

    content.append(Paragraph("CARDIOLOGY CONSULTATION REPORT", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Patient Name: ______________________________", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Date: {datetime.date.today()}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Patient Clinical Data", styles["Heading3"]))

    # Human readable values
    readable_data = [
        ["Age", f"{data['age']} years"],
        ["Biological Sex", "Male" if data["sex"] == 1 else "Female"],
        ["Chest Pain Type", list(cp_map.keys())[list(cp_map.values()).index(data["cp"])],
        ["Resting Blood Pressure", f"{data['trestbps']} millimeters of mercury"],
        ["Cholesterol", f"{data['chol']} milligrams per deciliter"],
        ["Fasting Blood Sugar", list(fbs_map.keys())[list(fbs_map.values()).index(data["fbs"])],
        ["Resting Electrocardiography", list(restecg_map.keys())[list(restecg_map.values()).index(data["restecg"])],
        ["Maximum Heart Rate", f"{data['thalach']} beats per minute"],
        ["Exercise Induced Angina", "Yes" if data["exang"] == 1 else "No"],
        ["ST Segment Depression", f"{data['oldpeak']} millimeters"],
        ["ST Segment Slope", list(slope_map.keys())[list(slope_map.values()).index(data["slope"])],
        ["Coronary Vessel Status", list(ca_map.keys())[list(ca_map.values()).index(data["ca"])],
        ["Thallium Test Result", list(thal_map.keys())[list(thal_map.values()).index(data["thal"])]
    ]

    # Wider table
    table = Table(
        [["Parameter", "Value"]] + readable_data,
        colWidths=[220, 300]
    )

    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))

    content.append(table)

    content.append(Spacer(1, 6))
    content.append(Paragraph("<b>Assessment</b>", small_style))
    content.append(Paragraph(
        f"Risk Category: {result['risk_category']} ({round(result['probability']*100,1)} percent)",
        small_style
    ))
    content.append(Paragraph(f"Credibility: {round(result['confidence']*100,1)} percent", small_style))

    content.append(Spacer(1, 6))
    content.append(Paragraph("<b>Clinical Interpretation</b>", small_style))
    content.append(Paragraph(result["clinical_reasoning"], small_style))

    content.append(Spacer(1, 6))
    content.append(Paragraph("<b>Management Plan</b>", small_style))

    for r in result["recommendations"]:
        content.append(Paragraph(f"- {r}", small_style))

    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "Recommendation: Referral to cardiology specialist is advised for further evaluation.",
        small_style
    ))

    content.append(Spacer(1, 12))
    content.append(Paragraph("Doctor Signature:", small_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph("______________________________", small_style))

    doc.build(content)
    buffer.seek(0)
    return buffer


    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    # smaller font style
    small_style = styles["Normal"].clone('small')
    small_style.fontSize = 8

    content = []

    content.append(Paragraph("CARDIOLOGY CONSULTATION REPORT", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Patient Name: ______________________________", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Date: {datetime.date.today()}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Patient Clinical Data", styles["Heading3"]))

    readable_data = [
        ["Age", data["age"]],
        ["Biological Sex", data["sex"]],
        ["Chest Pain Type", data["cp"]],
        ["Resting Blood Pressure", data["trestbps"]],
        ["Cholesterol", data["chol"]],
        ["Fasting Blood Sugar", data["fbs"]],
        ["Resting Electrocardiography", data["restecg"]],
        ["Maximum Heart Rate", data["thalach"]],
        ["Exercise Induced Angina", data["exang"]],
        ["ST Segment Depression", data["oldpeak"]],
        ["ST Segment Slope", data["slope"]],
        ["Coronary Vessel Status", data["ca"]],
        ["Thallium Test Result", data["thal"]]
    ]

    table = Table([["Parameter","Value"]] + readable_data)
    table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 1, colors.black)]))

    content.append(table)

    content.append(Spacer(1, 6))
    content.append(Paragraph("<b>Assessment</b>", small_style))
    content.append(Paragraph(
        f"Risk Category: {result['risk_category']} ({round(result['probability']*100,1)}%)",
        small_style
    ))
    content.append(Paragraph(f"Credibility: {round(result['confidence']*100,1)}%", small_style))

    content.append(Spacer(1, 6))
    content.append(Paragraph("<b>Clinical Interpretation</b>", small_style))
    content.append(Paragraph(result["clinical_reasoning"], small_style))

    content.append(Spacer(1, 6))
    content.append(Paragraph("<b>Management Plan</b>", small_style))

    for r in result["recommendations"]:
        content.append(Paragraph(f"- {r}", small_style))

    content.append(Spacer(1, 6))
    content.append(Paragraph(
        "Recommendation: Referral to cardiology specialist is advised for further evaluation.",
        small_style
    ))

    content.append(Spacer(1, 12))
    content.append(Paragraph("Doctor Signature:", small_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph("______________________________", small_style))

    doc.build(content)
    buffer.seek(0)
    return buffer

# =========================
# ANALYSIS
# =========================
st.divider()

if st.button("🧠 Run Clinical Risk Analysis", use_container_width=True):

  with st.spinner("The AI is Thinking..."):
    try:
        API_URL = "https://cardio-ai-msud.onrender.com/predict"

        response = requests.post(
            API_URL,
            json=data,
            timeout=100
        )

        result = response.json()

    except Exception as e:
        st.error(f"Backend connection failed: {e}")
        st.stop()

        

    result["recommendations"] = improve_recommendations(result["recommendations"])

    prob = result["probability"]
    confidence = result["confidence"]

    st.header("📊 Clinical Risk Assessment")

    risk = result["risk_category"]

    # Bigger font risk category
    if "Low" in risk:
        st.markdown(f"<h3 style='color:green'>Risk Category: {risk}</h3>", unsafe_allow_html=True)
    elif "Intermediate" in risk:
        st.markdown(f"<h3 style='color:orange'>Risk Category: {risk}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red'>Risk Category: {risk}</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability", f"{prob*100:.1f}%")
        st.progress(prob)

    with col2:
        st.metric("Credibility", f"{confidence*100:.1f}%")
        st.progress(confidence)

    st.divider()
    st.subheader("🧠 Clinical Interpretation")
    st.info(result["clinical_reasoning"])

    st.divider()
    st.subheader("💡 Clinical Recommendations")

    for r in result["recommendations"]:
        st.success(r)
        st.markdown(f"[Learn more](https://www.google.com/search?q={r.replace(' ', '+')})")

    st.divider()
    st.subheader("🧾 Cardiology Consultation Report")

    pdf = generate_pdf(result, data)

    st.download_button(
        "⬇️ Download Report",
        data=pdf,
        file_name="cardiology_report.pdf",
        mime="application/pdf"
    )
