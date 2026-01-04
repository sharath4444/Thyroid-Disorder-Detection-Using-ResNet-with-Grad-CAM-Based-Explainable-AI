import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pickle
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import io

# Page configuration
st.set_page_config(
    page_title="Thyroid Disorder Detection",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CONFIG ---------------- 
DL_MODEL_PATH = "new_thyroid_resnet50_best.h5"
ML_MODEL_PATH = "ml_model.pkl"
ML_SCALER_PATH = "ml_scaler.pkl"
IMG_SIZE = 224
DL_CLASS_NAMES = ["Benign", "Malignant"]
LAST_CONV = "conv5_block3_out"

# ---------------- LOAD MODELS ---------------- 
@st.cache_resource
def load_dl_model():
    """Load the deep learning model"""
    return tf.keras.models.load_model(DL_MODEL_PATH)

@st.cache_resource
def load_ml_model():
    """Load the machine learning model and scaler"""
    with open(ML_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ML_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load models
try:
    dl_model = load_dl_model()
    ml_model, ml_scaler = load_ml_model()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# ---------------- GRAD-CAM FUNCTION ---------------- 
def gradcam(img_array, model):
    """Generate Grad-CAM heatmap for deep learning model"""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        # Handle single-output & multi-output models
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        cls = tf.argmax(preds[0])
        loss = preds[:, cls]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    confidence = float(tf.reduce_max(preds))
    return heatmap.numpy(), int(cls.numpy()), confidence

# ---------------- DL PREDICTION FUNCTION ---------------- 
def predict_dl(image):
    """Predict using deep learning model"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR if needed, then back to RGB
    if len(img_array.shape) == 3:
        orig = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    else:
        orig = img_array
    
    # Resize and preprocess
    img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    x = preprocess_input(np.expand_dims(img, 0))
    
    # Get prediction and Grad-CAM
    heatmap, cls, conf = gradcam(x, dl_model)
    conf *= 100
    
    # Create overlay
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)
    
    return {
        'prediction': DL_CLASS_NAMES[cls],
        'confidence': conf,
        'overlay_image': overlay,
        'original_image': orig,
        'heatmap': heatmap_resized
    }

# ---------------- ML PREDICTION FUNCTION ---------------- 
def predict_ml(age, sex, tsh, t3, t4, t4u, fti):
    """Predict using machine learning model"""
    # Encode sex
    sex_encoded = 0 if sex == "M" else 1
    
    # Create input dataframe
    input_data = pd.DataFrame([[age, sex_encoded, tsh, t3, t4, t4u, fti]],
                              columns=['age', 'sex', 'TSH', 'T3', 'T4', 'T4U', 'FTI'])
    
    # Scale input
    input_scaled = ml_scaler.transform(input_data)
    
    # Predict
    prediction = ml_model.predict(input_scaled)[0]
    prediction_proba = ml_model.predict_proba(input_scaled)[0]
    
    # Get confidence (max probability)
    confidence = max(prediction_proba) * 100
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': dict(zip(ml_model.classes_, prediction_proba))
    }

# ---------------- FUSION FUNCTION ---------------- 
def fuse_predictions(dl_result, ml_result):
    """Combine DL and ML predictions to get final 3-label output"""
    # Map ML "negative" to "normal"
    ml_pred = ml_result['prediction']
    if ml_pred == "negative":
        ml_pred = "normal"
    
    # Calculate confidences
    dl_conf = dl_result['confidence'] / 100
    ml_conf = ml_result['confidence'] / 100
    
    # Weighted fusion (can be adjusted)
    dl_weight = 0.6  # 60% weight for DL
    ml_weight = 0.4  # 40% weight for ML
    
    fused_confidence = (dl_conf * dl_weight + ml_conf * ml_weight) * 100
    
    # Final prediction: Use ML prediction as primary (since it has the 3 categories)
    # DL prediction (Benign/Malignant) is used for validation
    final_prediction = ml_pred  # hyperthyroid, hypothyroid, or normal
    
    # Map ML probabilities
    ml_probs = ml_result['probabilities'].copy()
    if 'negative' in ml_probs:
        ml_probs['normal'] = ml_probs.pop('negative')
    
    return {
        'final_prediction': final_prediction,  # hyperthyroid, hypothyroid, or normal
        'final_confidence': fused_confidence,
        'dl_prediction': dl_result['prediction'],
        'dl_confidence': dl_result['confidence'],
        'ml_prediction': ml_result['prediction'],
        'ml_confidence': ml_result['confidence'],
        'ml_probabilities': ml_probs
    }

# ---------------- STREAMLIT UI ---------------- 
st.title("🏥 Thyroid Disorder Detection System")
st.markdown("### Combined Deep Learning & Machine Learning Analysis")

if not models_loaded:
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Analysis Type", 
                       ["Deep Learning (Image)", "Machine Learning (Blood Test)", "Combined Analysis"])

# ---------------- DEEP LEARNING PAGE ---------------- 
if page == "Deep Learning (Image)":
    st.header("🔬 Deep Learning Analysis (ResNet50 + Grad-CAM)")
    st.markdown("Upload a thyroid ultrasound image for analysis")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Processing image..."):
                try:
                    dl_result = predict_dl(image)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(dl_result['original_image'], use_container_width=True)
                    
                    with col2:
                        st.subheader("Grad-CAM Visualization")
                        st.image(dl_result['overlay_image'], use_container_width=True)
                    
                    st.success(f"**Prediction:** {dl_result['prediction']}")
                    st.info(f"**Confidence:** {dl_result['confidence']:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# ---------------- MACHINE LEARNING PAGE ---------------- 
elif page == "Machine Learning (Blood Test)":
    st.header("🧪 Machine Learning Analysis (Random Forest)")
    st.markdown("Enter blood test parameters for analysis")
    
    # Reference table
    with st.expander("📋 Reference Ranges for Blood Test Parameters"):
        reference_data = {
            "Parameter": ["Age", "Sex", "TSH (µIU/mL)", "T3 (ng/dL)", "T4 (µg/dL)", "T4U", "FTI"],
            "Normal Range": ["1-80 years", "M/F", "0.4 - 4.0", "80 - 200", "4.5 - 12.0", "0.5 - 2.0", "4 - 12"]
        }
        ref_df = pd.DataFrame(reference_data)
        st.table(ref_df)
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=80, value=30)
        sex = st.selectbox("Sex", options=["M", "F"])
        tsh = st.number_input("TSH (µIU/mL)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
        t3 = st.number_input("T3 (ng/dL)", min_value=50.0, max_value=400.0, value=100.0, step=0.1)
    
    with col2:
        t4 = st.number_input("T4 (µg/dL)", min_value=2.0, max_value=25.0, value=8.0, step=0.1)
        t4u = st.number_input("T4U", min_value=0.5, max_value=2.5, value=1.0, step=0.1)
        fti = st.number_input("FTI", min_value=2.0, max_value=300.0, value=8.0, step=0.1)
    
    if st.button("Predict"):
        with st.spinner("Analyzing blood test results..."):
            try:
                ml_result = predict_ml(age, sex, tsh, t3, t4, t4u, fti)
                
                # Map "negative" to "normal"
                final_pred = ml_result['prediction']
                if final_pred == "negative":
                    final_pred = "normal"
                
                # Color coding
                if final_pred == "hyperthyroid":
                    pred_emoji = "🔴"
                    pred_color = "#d32f2f"
                elif final_pred == "hypothyroid":
                    pred_emoji = "🟡"
                    pred_color = "#f57c00"
                else:
                    pred_emoji = "🟢"
                    pred_color = "#388e3c"
                
                # Display result prominently
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 20px 0;">
                    <h1 style="color: white; margin-bottom: 10px;">{pred_emoji} FINAL RESULT</h1>
                    <h2 style="color: {pred_color}; font-size: 2em; font-weight: bold; margin: 10px 0;">{final_pred.upper()}</h2>
                    <p style="color: white; font-size: 1.2em; margin-top: 10px;">Confidence: {ml_result['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 View Detailed Probabilities", expanded=False):
                    prob_dict = ml_result['probabilities'].copy()
                    if 'negative' in prob_dict:
                        prob_dict['normal'] = prob_dict.pop('negative')
                    prob_df = pd.DataFrame([prob_dict])
                    st.dataframe(prob_df, use_container_width=True)
                    st.bar_chart(prob_df.T)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ---------------- COMBINED ANALYSIS PAGE ---------------- 
elif page == "Combined Analysis":
    st.header("🔬🧪 Combined Deep Learning & Machine Learning Analysis")
    st.markdown("Get comprehensive analysis using both image and blood test data")
    
    # Initialize session state variables
    if 'dl_result' not in st.session_state:
        st.session_state.dl_result = None
    if 'ml_result' not in st.session_state:
        st.session_state.ml_result = None
    if 'dl_analyzed' not in st.session_state:
        st.session_state.dl_analyzed = False
    if 'ml_analyzed' not in st.session_state:
        st.session_state.ml_analyzed = False
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    # Two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Image Analysis")
        uploaded_file = st.file_uploader("Upload Thyroid Image", type=['jpg', 'jpeg', 'png'], key="dl_upload")
        
        # Reset DL results if file changes
        if uploaded_file is not None:
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.dl_result = None
                st.session_state.dl_analyzed = False
                st.session_state.last_uploaded_file = uploaded_file.name
            
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Show existing DL result if available
            if st.session_state.dl_analyzed and st.session_state.dl_result:
                st.success(f"✅ DL Prediction: {st.session_state.dl_result['prediction']} ({st.session_state.dl_result['confidence']:.2f}%)")
            
            if st.button("Analyze Image", key="dl_btn"):
                with st.spinner("Processing image..."):
                    try:
                        dl_result = predict_dl(image)
                        st.session_state.dl_result = dl_result
                        st.session_state.dl_analyzed = True
                        st.success(f"✅ DL Prediction: {dl_result['prediction']} ({dl_result['confidence']:.2f}%)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            # Reset if file is removed
            if st.session_state.dl_analyzed:
                st.session_state.dl_result = None
                st.session_state.dl_analyzed = False
                st.session_state.last_uploaded_file = None
    
    with col2:
        st.subheader("🧪 Blood Test Analysis")
        age = st.number_input("Age (years)", min_value=1, max_value=80, value=30, key="ml_age")
        sex = st.selectbox("Sex", options=["M", "F"], key="ml_sex")
        tsh = st.number_input("TSH (µIU/mL)", min_value=0.1, max_value=50.0, value=2.0, step=0.1, key="ml_tsh")
        t3 = st.number_input("T3 (ng/dL)", min_value=50.0, max_value=400.0, value=100.0, step=0.1, key="ml_t3")
        t4 = st.number_input("T4 (µg/dL)", min_value=2.0, max_value=25.0, value=8.0, step=0.1, key="ml_t4")
        t4u = st.number_input("T4U", min_value=0.5, max_value=2.5, value=1.0, step=0.1, key="ml_t4u")
        fti = st.number_input("FTI", min_value=2.0, max_value=300.0, value=8.0, step=0.1, key="ml_fti")
        
        # Show existing ML result if available
        if st.session_state.ml_analyzed and st.session_state.ml_result:
            ml_pred_display = st.session_state.ml_result['prediction']
            if ml_pred_display == "negative":
                ml_pred_display = "normal"
            st.success(f"✅ ML Prediction: {ml_pred_display.upper()} ({st.session_state.ml_result['confidence']:.2f}%)")
        
        if st.button("Analyze Blood Test", key="ml_btn"):
            with st.spinner("Analyzing blood test..."):
                try:
                    ml_result = predict_ml(age, sex, tsh, t3, t4, t4u, fti)
                    st.session_state.ml_result = ml_result
                    st.session_state.ml_analyzed = True
                    ml_pred_display = ml_result['prediction']
                    if ml_pred_display == "negative":
                        ml_pred_display = "normal"
                    st.success(f"✅ ML Prediction: {ml_pred_display.upper()} ({ml_result['confidence']:.2f}%)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Combined Results Section
    if st.session_state.dl_analyzed and st.session_state.ml_analyzed:
        st.markdown("---")
        st.header("📊 Final Diagnosis Result")
        
        # Fuse predictions
        fused_result = fuse_predictions(st.session_state.dl_result, st.session_state.ml_result)
        
        # Display final prediction prominently
        final_pred = fused_result['final_prediction'].upper()
        
        # Color coding based on prediction
        if final_pred == "HYPERTHYROID":
            pred_color = "🔴"
            pred_style = "color: #d32f2f; font-size: 2.5em; font-weight: bold;"
        elif final_pred == "HYPOTHYROID":
            pred_color = "🟡"
            pred_style = "color: #f57c00; font-size: 2.5em; font-weight: bold;"
        else:  # NORMAL
            pred_color = "🟢"
            pred_style = "color: #388e3c; font-size: 2.5em; font-weight: bold;"
        
        # Main result display
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 20px 0;">
            <h1 style="color: white; margin-bottom: 10px;">{pred_color} FINAL RESULT</h1>
            <h2 style="{pred_style} margin: 20px 0;">{final_pred}</h2>
            <p style="color: white; font-size: 1.2em; margin-top: 10px;">Confidence: {fused_result['final_confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown (collapsible)
        with st.expander("📋 Detailed Analysis Breakdown", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("DL Prediction", fused_result['dl_prediction'])
                st.caption(f"Confidence: {fused_result['dl_confidence']:.2f}%")
            
            with col2:
                ml_pred_display = fused_result['ml_prediction']
                if ml_pred_display == "negative":
                    ml_pred_display = "normal"
                st.metric("ML Prediction", ml_pred_display)
                st.caption(f"Confidence: {fused_result['ml_confidence']:.2f}%")
            
            with col3:
                st.metric("Combined Confidence", f"{fused_result['final_confidence']:.2f}%")
        
        # Show Grad-CAM visualization
        if st.session_state.dl_result:
            st.subheader("🔬 Image Analysis Visualization")
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.dl_result['original_image'], caption="Original Image", use_container_width=True)
            with col2:
                st.image(st.session_state.dl_result['overlay_image'], caption="Grad-CAM Overlay", use_container_width=True)
        
        # Show ML probabilities (optional)
        with st.expander("📊 Probability Distribution", expanded=False):
            prob_df = pd.DataFrame([fused_result['ml_probabilities']])
            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(prob_df.T)
        
        # Final recommendation
        st.markdown("---")
        st.info("⚠️ **Important:** This result is for research purposes only. Please consult with a healthcare professional for a comprehensive medical diagnosis.")
        
    elif st.session_state.dl_analyzed or st.session_state.ml_analyzed:
        st.markdown("---")
        st.warning("⚠️ Please complete both analyses to see final diagnosis result.")
        
        # Show individual results if available
        if st.session_state.ml_analyzed and st.session_state.ml_result:
            st.subheader("🧪 Machine Learning Results")
            ml_pred_display = st.session_state.ml_result['prediction']
            if ml_pred_display == "negative":
                ml_pred_display = "normal"
            
            # Display with color coding
            if ml_pred_display == "hyperthyroid":
                pred_emoji = "🔴"
            elif ml_pred_display == "hypothyroid":
                pred_emoji = "🟡"
            else:
                pred_emoji = "🟢"
            
            st.markdown(f"### {pred_emoji} **Prediction: {ml_pred_display.upper()}**")
            st.info(f"**Confidence:** {st.session_state.ml_result['confidence']:.2f}%")
            
            with st.expander("📊 View Probabilities"):
                prob_dict = st.session_state.ml_result['probabilities'].copy()
                if 'negative' in prob_dict:
                    prob_dict['normal'] = prob_dict.pop('negative')
                prob_df = pd.DataFrame([prob_dict])
                st.dataframe(prob_df, use_container_width=True)
                st.bar_chart(prob_df.T)
        
        if st.session_state.dl_analyzed and st.session_state.dl_result:
            st.subheader("🔬 Deep Learning Results")
            st.success(f"**Image Analysis:** {st.session_state.dl_result['prediction']}")
            st.info(f"**Confidence:** {st.session_state.dl_result['confidence']:.2f}%")
            st.caption("Note: Complete both analyses to get final diagnosis (hyperthyroid/hypothyroid/normal)")
    else:
        st.info("👆 Please upload an image and enter blood test values, then click the analyze buttons to see combined results.")
    
    # Add clear button
    if st.session_state.dl_analyzed or st.session_state.ml_analyzed:
        st.markdown("---")
        if st.button("🔄 Clear All Results", key="clear_btn"):
            st.session_state.dl_result = None
            st.session_state.ml_result = None
            st.session_state.dl_analyzed = False
            st.session_state.ml_analyzed = False
            st.session_state.last_uploaded_file = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**Note:** This tool is for research purposes only. Always consult with healthcare professionals for medical diagnosis.")
