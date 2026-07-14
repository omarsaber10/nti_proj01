import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# Custom CSS - website-like styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page */
    .main {
        background-color: #f7f9fc;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(79, 172, 254, 0.25);
    }
    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.4rem;
    }
    .hero p {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0;
    }

    /* Card container */
    .form-card {
        background: white;
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border: 1px solid #eef1f6;
    }

    .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1f3b57;
        margin-bottom: 0.6rem;
    }

    /* Result cards by risk level */
    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        margin-top: 1rem;
        text-align: center;
        border: 2px solid;
    }
    .risk-low {
        background-color: #eafaf1;
        border-color: #2ecc71;
        color: #1e7e45;
    }
    .risk-moderate {
        background-color: #fff8e6;
        border-color: #f5b942;
        color: #8a6100;
    }
    .risk-high {
        background-color: #fdecea;
        border-color: #e74c3c;
        color: #a3281c;
    }
    .risk-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .risk-sub {
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Predict button */
    div.stButton > button {
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        font-size: 1.05rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model and scaler (cached so they only load once)
# ------------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = keras.models.load_model("Pima Indians Diabetes.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🩺 Pima Indians Diabetes Prediction</h1>
    <p>أدخل البيانات الطبية للمريض وسيقوم النموذج بتقدير نسبة خطورة الإصابة بمرض السكري</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 بيانات المريض</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("🔍 تنبؤ الآن", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if predict_clicked:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    # Scale using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Predict probability
    prob = float(model.predict(input_scaled, verbose=0)[0][0])
    pct = prob * 100

    # ------------------------------------------------------------------
    # Risk level tied to the actual probability, not just a binary cutoff
    # ------------------------------------------------------------------
    if pct < 30:
        risk_class = "risk-low"
        icon = "✅"
        title = "خطورة منخفضة"
        sub = "المؤشرات الحالية لا تدل على وجود مخاطر كبيرة، مع ذلك يُفضّل المتابعة الدورية."
    elif pct < 50:
        risk_class = "risk-moderate"
        icon = "🟡"
        title = "خطورة منخفضة إلى متوسطة"
        sub = "هناك بعض المؤشرات التي تستحق الانتباه، يُنصح بمتابعة نمط الحياة الصحي."
    elif pct < 70:
        risk_class = "risk-moderate"
        icon = "⚠️"
        title = "خطورة متوسطة"
        sub = "توجد مؤشرات ملحوظة على احتمال الإصابة، يُفضّل استشارة الطبيب."
    elif pct < 85:
        risk_class = "risk-high"
        icon = "🔶"
        title = "خطورة عالية"
        sub = "احتمالية الإصابة مرتفعة، يُنصح بشدة بمراجعة الطبيب لإجراء فحوصات إضافية."
    else:
        risk_class = "risk-high"
        icon = "🔴"
        title = "خطورة عالية جداً"
        sub = "المؤشرات تدل بقوة على احتمال الإصابة بمرض السكري، يجب مراجعة الطبيب فوراً."

    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div class="risk-title">{icon} {title}</div>
        <div class="risk-sub">{sub}</div>
        <div style="margin-top:0.7rem; font-size:1.6rem; font-weight:800;">{pct:.1f}%</div>
        <div class="risk-sub">نسبة احتمال الإصابة بالسكري</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(prob)
    st.caption(f"القيمة الاحتمالية الدقيقة: {prob:.4f}")

st.divider()
st.caption(
    "Model: Keras Sequential neural network trained on the Pima Indians "
    "Diabetes dataset. Features are standardized with the same StandardScaler "
    "used during training."
)