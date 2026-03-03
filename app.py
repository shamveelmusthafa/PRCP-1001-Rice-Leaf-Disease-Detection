import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import os
import time
from datetime import datetime

st.set_page_config(page_title="Rice Guard", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&display=swap");
:root{--green-dark:#0a1a0a;--green-mid:#1a3a1a;--green-light:#a8d878;--green-muted:#7aab5a;--gold:#f0c040;--red:#ff6b6b;}
*{transition:all 0.25s ease;}
.stApp{background:radial-gradient(ellipse at top left,#1a3a0a 0%,#0a1a0a 40%,#0d1f0d 100%);font-family:"DM Sans",sans-serif;}
.hero{text-align:center;padding:3rem 0 1rem 0;}
.hero-badge{display:inline-block;background:rgba(168,216,120,0.12);border:1px solid rgba(168,216,120,0.3);border-radius:50px;padding:6px 20px;font-size:0.75rem;letter-spacing:4px;color:#7aab5a;text-transform:uppercase;margin-bottom:1rem;}
.main-title{font-family:"Playfair Display",serif;font-size:clamp(2.5rem,5vw,4rem);font-weight:900;background:linear-gradient(135deg,#a8d878,#f0c040,#a8d878);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-1px;line-height:1.1;margin:0;}
.sub-title{color:#7aab5a;font-size:1rem;letter-spacing:3px;text-transform:uppercase;margin-top:0.5rem;}
.result-card{background:linear-gradient(145deg,rgba(26,58,26,0.9),rgba(13,42,13,1));border:1px solid rgba(168,216,120,0.2);border-radius:20px;padding:28px;margin:12px 0;box-shadow:0 8px 32px rgba(0,0,0,0.4);animation:slideUp 0.5s ease;}
.rejected-card{background:linear-gradient(145deg,rgba(58,26,26,0.9),rgba(42,13,13,1));border:1px solid rgba(255,107,107,0.3);border-radius:20px;padding:28px;margin:12px 0;animation:slideUp 0.5s ease;}
.disease-name{font-family:"Playfair Display",serif;font-size:2.2rem;font-weight:900;margin:0 0 8px 0;line-height:1.2;}
.confidence-badge{display:inline-block;background:rgba(168,216,120,0.15);border:1px solid rgba(168,216,120,0.3);border-radius:50px;padding:4px 16px;font-size:0.9rem;color:#a8d878;margin-top:8px;}
.severity-high{background:rgba(255,100,50,0.15);border:1px solid rgba(255,100,50,0.4);color:#ff6432;border-radius:50px;padding:3px 12px;font-size:0.8rem;display:inline-block;}
.severity-med{background:rgba(255,159,67,0.15);border:1px solid rgba(255,159,67,0.4);color:#ff9f43;border-radius:50px;padding:3px 12px;font-size:0.8rem;display:inline-block;}
.severity-low{background:rgba(144,238,144,0.15);border:1px solid rgba(144,238,144,0.4);color:#90ee90;border-radius:50px;padding:3px 12px;font-size:0.8rem;display:inline-block;}
.info-box{background:rgba(168,216,120,0.05);border-left:3px solid #a8d878;border-radius:0 12px 12px 0;padding:16px 20px;margin:10px 0;color:#c8e8a8;font-size:0.92rem;line-height:1.7;}
.treatment-box{background:rgba(120,200,120,0.05);border-left:3px solid #78c878;border-radius:0 12px 12px 0;padding:16px 20px;margin:10px 0;color:#c8e8a8;font-size:0.92rem;line-height:1.7;}
.section-header{font-family:"Playfair Display",serif;font-size:1.3rem;color:#a8d878;border-bottom:1px solid rgba(168,216,120,0.2);padding-bottom:10px;margin:24px 0 16px 0;display:flex;align-items:center;gap:10px;}
.upload-hint{text-align:center;padding:40px 20px;border:2px dashed rgba(168,216,120,0.2);border-radius:20px;background:rgba(168,216,120,0.02);margin:16px 0;}
.stButton>button{background:linear-gradient(135deg,#2a5a0a,#4a7a1a);color:#e8f8d0;border:1px solid rgba(168,216,120,0.3);border-radius:10px;padding:10px 24px;font-family:"DM Sans",sans-serif;font-size:0.9rem;width:100%;box-shadow:0 4px 15px rgba(0,0,0,0.3);}
.stButton>button:hover{background:linear-gradient(135deg,#4a7a1a,#6a9a2a);transform:translateY(-2px);}
.clear-btn>button{background:linear-gradient(135deg,#5a1a1a,#7a2a2a) !important;border-color:rgba(255,107,107,0.3) !important;color:#ffd0d0 !important;}
.clear-btn>button:hover{background:linear-gradient(135deg,#7a2a2a,#9a3a3a) !important;}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#050f05 0%,#0d1f0d 100%);border-right:1px solid rgba(168,216,120,0.1);}
section[data-testid="stSidebarContent"]{padding:2rem 1.2rem;}
.stTabs [data-baseweb="tab-list"]{background:rgba(168,216,120,0.04);border-radius:12px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:10px;color:#7aab5a;font-family:"DM Sans",sans-serif;font-size:0.9rem;padding:10px 24px;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#2a5a0a,#4a7a1a) !important;color:#e8f8d0 !important;box-shadow:0 4px 12px rgba(0,0,0,0.3);}
.prob-row{margin:8px 0;}
.prob-label{display:flex;justify-content:space-between;color:#c8e8a8;font-size:0.85rem;margin-bottom:4px;}
.prob-bar-bg{background:rgba(168,216,120,0.1);border-radius:50px;height:8px;overflow:hidden;}
.prob-bar-fill{height:100%;border-radius:50px;}
.gradcam-label{text-align:center;color:#a8d878;font-weight:600;font-size:0.9rem;margin-bottom:8px;padding:6px;background:rgba(168,216,120,0.06);border-radius:8px;}
.timing-badge{display:inline-block;background:rgba(240,192,64,0.1);border:1px solid rgba(240,192,64,0.3);border-radius:50px;padding:4px 14px;font-size:0.82rem;color:#f0c040;margin-top:6px;}
.loading-container{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;gap:16px;border:2px dashed rgba(168,216,120,0.15);border-radius:20px;background:rgba(168,216,120,0.02);}
.leaf-spinner{font-size:3.5rem;animation:spin 1.2s linear infinite;display:inline-block;}
.loading-text{color:#a8d878;font-family:"Playfair Display",serif;font-size:1.2rem;animation:pulse 1.5s ease infinite;}
.loading-sub{color:#7aab5a;font-size:0.78rem;letter-spacing:3px;text-transform:uppercase;}
@keyframes spin{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#0a1a0a;}
::-webkit-scrollbar-thumb{background:#3a6a2a;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

@st.cache_resource(show_spinner=False)
def load_disease_model():
    try:
        model = load_model("rice_disease_best_model.keras")
        return model
    except Exception as e:
        st.error(f"Model not found! Make sure rice_disease_best_model.keras is in the same folder. Error: {e}")
        return None

CLASS_INFO = {
    "Bacterial leaf blight": {
        "description": "A serious bacterial disease caused by Xanthomonas oryzae pv. oryzae. Water-soaked lesions appear along leaf margins, turning yellow then white as they expand.",
        "treatment": "Apply copper-based bactericides. Use disease-resistant varieties. Ensure proper drainage and avoid excessive nitrogen fertilizer.",
        "severity": "High", "severity_class": "severity-high",
        "color": "#ff9f43", "icon": "🔴",
        "facts": ["Spreads rapidly in wet conditions", "Can reduce yield by 20-30%", "Most severe during tillering stage"]
    },
    "Brown spot": {
        "description": "A fungal disease caused by Cochliobolus miyabeanus. Dark brown oval spots with yellow halos appear scattered across the leaf surface.",
        "treatment": "Apply Mancozeb or Iprobenfos fungicide. Maintain balanced soil nutrition especially potassium and silicon.",
        "severity": "Medium", "severity_class": "severity-med",
        "color": "#cd853f", "icon": "🟠",
        "facts": ["Favored by drought stress", "Indicates nutrient deficiency", "Can affect grain quality"]
    },
    "Leaf smut": {
        "description": "A fungal disease caused by Entyloma oryzae. Small angular black spots appear on both sides of the leaf blade.",
        "treatment": "Apply propiconazole fungicide. Use disease-free seeds and clean field practices.",
        "severity": "Low-Medium", "severity_class": "severity-low",
        "color": "#90ee90", "icon": "🟡",
        "facts": ["Usually mild impact on yield", "More common in humid regions", "Manageable with early detection"]
    }
}

CLASS_INDICES = {"Bacterial leaf blight": 0, "Brown spot": 1, "Leaf smut": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}

def generate_gradcam(model, img_array, predicted_idx):
    try:
        grad_model = Model(inputs=model.inputs, outputs=[model.get_layer("mixed10").output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(np.expand_dims(img_array, axis=0))
            loss = preds[:, predicted_idx]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap.numpy(), (200, 200))
        img_bgr = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.cvtColor(cv2.addWeighted(img_bgr, 0.6, colored, 0.4, 0), cv2.COLOR_BGR2RGB)
        return heatmap, overlay
    except:
        return None, None

def predict(image, model):
    arr = img_to_array(image.resize((200, 200))) / 255.0
    start_time = time.time()
    preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)
    elapsed = time.time() - start_time
    idx = np.argmax(preds[0])
    conf = preds[0][idx] * 100
    probs = {IDX_TO_CLASS[i]: float(preds[0][i]) * 100 for i in range(3)}
    return IDX_TO_CLASS[idx], conf, probs, arr, idx, elapsed

def clear_image():
    st.session_state.current_image = None
    st.session_state.upload_key += 1

def prob_bar(label, prob, color):
    st.markdown(f"""
    <div class="prob-row">
        <div class="prob-label"><span>{label}</span><span>{prob:.1f}%</span></div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{prob}%;background:linear-gradient(90deg,{color}88,{color});"></div>
        </div>
    </div>""", unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:
    st.markdown("<div style='font-family:Playfair Display,serif;color:#a8d878;font-size:1.4rem;font-weight:700;margin-bottom:4px;'>Settings</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7aab5a;font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;'>Configuration Panel</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#a8d878;font-weight:600;font-size:0.85rem;margin-bottom:10px;'>CONFIDENCE THRESHOLD</div>", unsafe_allow_html=True)
    threshold = st.slider("", min_value=0.50, max_value=0.95, value=0.70, step=0.05, key="threshold_slider")
    st.markdown(f"<div style='background:rgba(168,216,120,0.06);border-radius:10px;padding:10px;text-align:center;'><span style='font-family:Playfair Display,serif;font-size:1.5rem;color:#a8d878;'>{int(threshold*100)}%</span><div style='color:#7aab5a;font-size:0.75rem;margin-top:2px;'>Minimum confidence to accept</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#a8d878;font-weight:600;font-size:0.85rem;margin-bottom:10px;'>VISUALIZATION OPTIONS</div>", unsafe_allow_html=True)
    show_gradcam = st.checkbox("GradCAM Heatmap", value=True)
    show_probs   = st.checkbox("Probability Bars", value=True)
    show_facts   = st.checkbox("Disease Facts",    value=True)
    st.markdown("---")
    st.markdown("<div style='color:#a8d878;font-weight:600;font-size:0.85rem;margin-bottom:12px;'>MODEL STATS</div>", unsafe_allow_html=True)
    st.markdown("""<div style='background:rgba(168,216,120,0.04);border-radius:12px;padding:14px;'>
<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(168,216,120,0.1);'><span style='color:#7aab5a;font-size:0.82rem;'>Model</span><span style='color:#a8d878;font-size:0.82rem;font-weight:600;'>InceptionV3</span></div>
<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(168,216,120,0.1);'><span style='color:#7aab5a;font-size:0.82rem;'>Accuracy</span><span style='color:#a8d878;font-size:0.82rem;font-weight:600;'>95.65%</span></div>
<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(168,216,120,0.1);'><span style='color:#7aab5a;font-size:0.82rem;'>Test Loss</span><span style='color:#a8d878;font-size:0.82rem;font-weight:600;'>0.1606</span></div>
<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(168,216,120,0.1);'><span style='color:#7aab5a;font-size:0.82rem;'>Training</span><span style='color:#a8d878;font-size:0.82rem;font-weight:600;'>Two Phase</span></div>
<div style='display:flex;justify-content:space-between;padding:5px 0;'><span style='color:#7aab5a;font-size:0.82rem;'>Input Size</span><span style='color:#a8d878;font-size:0.82rem;font-weight:600;'>200x200px</span></div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    total    = len(st.session_state.prediction_history)
    accepted = sum(1 for h in st.session_state.prediction_history if "Rejected" not in h["disease"])
    rejected = total - accepted
    st.markdown("<div style='color:#a8d878;font-weight:600;font-size:0.85rem;margin-bottom:12px;'>SESSION STATS</div>", unsafe_allow_html=True)
    st.markdown(f"""<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;'>
<div style='background:rgba(168,216,120,0.06);border-radius:10px;padding:10px;text-align:center;'><div style='font-family:Playfair Display,serif;font-size:1.4rem;color:#a8d878;'>{total}</div><div style='color:#7aab5a;font-size:0.7rem;'>Total</div></div>
<div style='background:rgba(168,216,120,0.06);border-radius:10px;padding:10px;text-align:center;'><div style='font-family:Playfair Display,serif;font-size:1.4rem;color:#90ee90;'>{accepted}</div><div style='color:#7aab5a;font-size:0.7rem;'>Accepted</div></div>
<div style='background:rgba(255,107,107,0.06);border-radius:10px;padding:10px;text-align:center;'><div style='font-family:Playfair Display,serif;font-size:1.4rem;color:#ff6b6b;'>{rejected}</div><div style='color:#7aab5a;font-size:0.7rem;'>Rejected</div></div>
</div>""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Diagnostics</div>
    <h1 class="main-title">Rice Guard</h1>
    <p class="sub-title">Plant Disease Detection System</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔬  Diagnose", "📋  History", "📚  Disease Library"])

# TAB 1
with tab1:
    upload_col, result_col = st.columns([1.1, 1], gap="large")

    with upload_col:
        st.markdown("<div class='section-header'>📤 Upload Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "", type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.upload_key}",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            st.session_state.current_image = uploaded_file

        if st.session_state.current_image is not None:
            image_display = Image.open(st.session_state.current_image).convert("RGB")
            st.image(image_display, use_container_width=True)
            cl1, cl2 = st.columns([1, 3])
            with cl1:
                st.markdown("<div class='clear-btn'>", unsafe_allow_html=True)
                if st.button("🗑️ Clear"):
                    clear_image()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="upload-hint">
            <div style="font-size:3rem;margin-bottom:12px;">🌿</div>
            <div style="font-family:Playfair Display,serif;color:#a8d878;font-size:1.3rem;margin-bottom:8px;">Drop your rice leaf image here</div>
            <div style="color:#7aab5a;font-size:0.85rem;line-height:1.8;">Supported: JPG · JPEG · PNG<br>Non-leaf images are automatically rejected</div>
            </div>""", unsafe_allow_html=True)

    with result_col:
        st.markdown("<div class='section-header'>🔬 Diagnosis</div>", unsafe_allow_html=True)

        if st.session_state.current_image is not None:
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
            <div class="loading-container">
                <div class="leaf-spinner">🌿</div>
                <div class="loading-text">Analyzing leaf patterns...</div>
                <div class="loading-sub">Running AI Model · Please wait</div>
            </div>""", unsafe_allow_html=True)

            model = load_disease_model()

            if model:
                image_predict = Image.open(st.session_state.current_image).convert("RGB")
                predicted_class, confidence, all_probs, img_array, predicted_idx, elapsed = predict(image_predict, model)

                loading_placeholder.empty()

                upload_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                image_name    = st.session_state.current_image.name
                status        = predicted_class if confidence >= threshold * 100 else "Rejected (Not a rice leaf)"
                already_saved = any(h["image_name"] == image_name for h in st.session_state.prediction_history)

                if not already_saved:
                    ib = io.BytesIO()
                    image_predict.save(ib, format="JPEG")
                    ib.seek(0)
                    rec_color = CLASS_INFO.get(predicted_class, {}).get("color", "#ff6b6b") if confidence >= threshold * 100 else "#ff6b6b"
                    st.session_state.prediction_history.insert(0, {
                        "image_name" : image_name,
                        "time"       : upload_time,
                        "disease"    : status,
                        "confidence" : f"{confidence:.2f}%",
                        "elapsed"    : f"{elapsed:.2f}s",
                        "color"      : rec_color,
                        "img_bytes"  : ib.getvalue()
                    })

                if confidence < threshold * 100:
                    st.markdown(f"""<div class="rejected-card">
                    <div style="font-size:2.5rem;margin-bottom:8px;">❌</div>
                    <div style="font-family:Playfair Display,serif;font-size:1.8rem;color:#ff6b6b;margin-bottom:8px;">Not a Rice Leaf</div>
                    <div style="color:#ff9f9f;font-size:0.9rem;">Confidence: {confidence:.1f}% is below {threshold*100:.0f}% threshold</div>
                    <div style="color:#cc7777;font-size:0.85rem;margin-top:8px;">Please upload a valid rice leaf image</div>
                    <div style="margin-top:12px;"><span class="timing-badge">Analyzed in {elapsed:.2f}s</span></div>
                    </div>""", unsafe_allow_html=True)
                else:
                    info = CLASS_INFO.get(predicted_class, {})
                    dc   = info.get("color", "#a8d878")
                    sc   = info.get("severity_class", "severity-low")
                    sv   = info.get("severity", "Unknown")
                    ic   = info.get("icon", "🌿")

                    st.markdown(f"""<div class="result-card">
                    <div style="font-size:2rem;margin-bottom:4px;">{ic}</div>
                    <div class="disease-name" style="color:{dc};">{predicted_class}</div>
                    <div style="margin:10px 0;display:flex;gap:8px;flex-wrap:wrap;">
                        <span class="confidence-badge">✓ {confidence:.1f}% Confidence</span>
                        <span class="{sc}">{sv} Severity</span>
                    </div>
                    <div><span class="timing-badge">⏱ Analyzed in {elapsed:.2f}s</span></div>
                    </div>""", unsafe_allow_html=True)

                    if show_probs:
                        st.markdown("<div class='section-header'>📊 Probabilities</div>", unsafe_allow_html=True)
                        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                            prob_bar(cls, prob, CLASS_INFO.get(cls, {}).get("color", "#a8d878"))

                    if show_facts and "facts" in info:
                        st.markdown("<div class='section-header'>💡 Key Facts</div>", unsafe_allow_html=True)
                        for fact in info["facts"]:
                            st.markdown(f"<div style='background:rgba(168,216,120,0.05);border-radius:8px;padding:8px 14px;margin:5px 0;color:#c8e8a8;font-size:0.85rem;'>• {fact}</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center;padding:60px 20px;border:2px dashed rgba(168,216,120,0.1);border-radius:20px;background:rgba(168,216,120,0.02);">
            <div style="font-size:3rem;margin-bottom:12px;opacity:0.4;">🔬</div>
            <div style="font-family:Playfair Display,serif;color:#5a8a4a;font-size:1.1rem;">Upload an image to begin</div>
            </div>""", unsafe_allow_html=True)

    # Disease info below image on left side
    if st.session_state.current_image is not None:
        model = load_disease_model()
        if model:
            image_info = Image.open(st.session_state.current_image).convert("RGB")
            pc, conf, _, _, _, _ = predict(image_info, model)
            if conf >= threshold * 100 and pc in CLASS_INFO:
                with upload_col:
                    info = CLASS_INFO[pc]
                    st.markdown("<div class='section-header'>ℹ️ About this Disease</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='info-box'><strong>About:</strong><br>{info['description']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='treatment-box'><strong>💊 Treatment:</strong><br>{info['treatment']}</div>", unsafe_allow_html=True)

    # GradCAM section
    if st.session_state.current_image is not None and show_gradcam:
        model = load_disease_model()
        if model:
            image_gc = Image.open(st.session_state.current_image).convert("RGB")
            pc, conf, _, img_array, pred_idx, _ = predict(image_gc, model)
            if conf >= threshold * 100:
                st.markdown("---")
                st.markdown("<div class='section-header'>🔍 GradCAM — Where did the model look?</div>", unsafe_allow_html=True)
                with st.spinner("Generating GradCAM heatmap..."):
                    heatmap, overlay = generate_gradcam(model, img_array, pred_idx)
                if heatmap is not None:
                    g1, g2, g3 = st.columns(3)
                    with g1:
                        st.markdown("<div class='gradcam-label'>🖼️ Original</div>", unsafe_allow_html=True)
                        st.image(image_gc.resize((200, 200)), use_container_width=True)
                    with g2:
                        st.markdown("<div class='gradcam-label'>🌡️ Heatmap</div>", unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(4, 4))
                        fig.patch.set_facecolor("#0a1a0a")
                        ax.imshow(heatmap, cmap="jet")
                        ax.axis("off")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0a1a0a", dpi=100)
                        buf.seek(0)
                        st.image(buf, use_container_width=True)
                        plt.close()
                    with g3:
                        st.markdown("<div class='gradcam-label'>🎯 Overlay</div>", unsafe_allow_html=True)
                        st.image(overlay, use_container_width=True)
                    st.markdown("<div class='info-box' style='margin-top:12px;'>🔴 <strong>Red/Yellow</strong> — High attention areas &nbsp;|&nbsp; 🔵 <strong>Blue</strong> — Low attention areas</div>", unsafe_allow_html=True)

# TAB 2
with tab2:
    st.markdown("<div class='section-header'>📋 Prediction History</div>", unsafe_allow_html=True)
    if len(st.session_state.prediction_history) == 0:
        st.markdown("""<div style="text-align:center;padding:50px;border:2px dashed rgba(168,216,120,0.2);border-radius:20px;background:rgba(168,216,120,0.02);">
        <div style="font-size:3rem;margin-bottom:12px;">📋</div>
        <div style="font-family:Playfair Display,serif;color:#a8d878;font-size:1.3rem;">No Predictions Yet</div>
        <div style="color:#7aab5a;font-size:0.9rem;margin-top:8px;">Upload images in the Diagnose tab to build history</div>
        </div>""", unsafe_allow_html=True)
    else:
        h1, h2, h3, h4 = st.columns([1, 1, 1, 3])
        with h1:
            st.markdown("<div class='clear-btn'>", unsafe_allow_html=True)
            if st.button("🗑️ Clear All"):
                st.session_state.prediction_history = []
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with h2:
            st.markdown(f"<div style='color:#a8d878;padding-top:10px;font-size:0.9rem;font-weight:600;'>{len(st.session_state.prediction_history)} Records</div>", unsafe_allow_html=True)
        with h3:
            accepted_count = sum(1 for h in st.session_state.prediction_history if "Rejected" not in h["disease"])
            st.markdown(f"<div style='color:#90ee90;padding-top:10px;font-size:0.9rem;'>✓ {accepted_count} Accepted</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='display:grid;grid-template-columns:70px 1.4fr 1.6fr 0.7fr 0.7fr 1.2fr;gap:10px;padding:10px 16px;background:rgba(168,216,120,0.08);border-radius:10px;margin-bottom:8px;'><div style='color:#a8d878;font-weight:600;font-size:0.82rem;'>Image</div><div style='color:#a8d878;font-weight:600;font-size:0.82rem;'>File Name</div><div style='color:#a8d878;font-weight:600;font-size:0.82rem;'>Disease</div><div style='color:#a8d878;font-weight:600;font-size:0.82rem;'>Confidence</div><div style='color:#a8d878;font-weight:600;font-size:0.82rem;'>Time Taken</div><div style='color:#a8d878;font-weight:600;font-size:0.82rem;'>Uploaded At</div></div>", unsafe_allow_html=True)

        for record in st.session_state.prediction_history:
            r_name    = record["image_name"]
            r_disease = record["disease"]
            r_color   = record["color"]
            r_conf    = record["confidence"]
            r_elapsed = record.get("elapsed", "N/A")
            r_time    = record["time"]

            ci, cn, cd, cc, ct, cu = st.columns([1, 2, 2, 1, 1, 2])
            with ci:
                thumb = Image.open(io.BytesIO(record["img_bytes"])).resize((55, 55))
                st.image(thumb)
            with cn:
                st.markdown(f"<div style='color:#c8e8a8;font-size:0.85rem;padding-top:8px;word-break:break-all;'>{r_name}</div>", unsafe_allow_html=True)
            with cd:
                st.markdown(f"<div style='color:{r_color};font-size:0.85rem;font-weight:600;padding-top:8px;'>{r_disease}</div>", unsafe_allow_html=True)
            with cc:
                st.markdown(f"<div style='color:#a8d878;font-size:0.85rem;padding-top:8px;'>{r_conf}</div>", unsafe_allow_html=True)
            with ct:
                st.markdown(f"<div style='color:#f0c040;font-size:0.85rem;padding-top:8px;'>{r_elapsed}</div>", unsafe_allow_html=True)
            with cu:
                st.markdown(f"<div style='color:#7aab5a;font-size:0.80rem;padding-top:8px;'>{r_time}</div>", unsafe_allow_html=True)
            st.markdown("<hr style='border-color:rgba(168,216,120,0.08);margin:4px 0;'>", unsafe_allow_html=True)

# TAB 3
with tab3:
    st.markdown("<div class='section-header'>📚 Rice Disease Library</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7aab5a;font-size:0.9rem;margin-bottom:20px;'>Learn about the three major rice diseases this system can detect</div>", unsafe_allow_html=True)
    for disease, info in CLASS_INFO.items():
        with st.expander(f"{info['icon']}  {disease}  —  {info['severity']} Severity", expanded=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                d_color = info["color"]
                d_sev   = info["severity"]
                d_sc    = info["severity_class"]
                d_icon  = info["icon"]
                st.markdown(f"""<div style='background:linear-gradient(145deg,rgba(26,58,26,0.8),rgba(13,42,13,0.9));border:1px solid {d_color}44;border-radius:16px;padding:24px;text-align:center;'>
                <div style='font-size:4rem;margin-bottom:8px;'>{d_icon}</div>
                <div style='font-family:Playfair Display,serif;color:{d_color};font-size:1.2rem;font-weight:700;'>{disease}</div>
                <div style='margin-top:10px;'><span class='{d_sc}'>{d_sev} Severity</span></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='info-box'><strong>📖 Description</strong><br><br>{info['description']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='treatment-box'><strong>💊 Treatment</strong><br><br>{info['treatment']}</div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-top:12px;'><strong style='color:#a8d878;font-size:0.85rem;'>💡 KEY FACTS</strong></div>", unsafe_allow_html=True)
                for fact in info["facts"]:
                    st.markdown(f"<div style='background:rgba(168,216,120,0.05);border-radius:8px;padding:8px 14px;margin:5px 0;color:#c8e8a8;font-size:0.85rem;'>• {fact}</div>", unsafe_allow_html=True)