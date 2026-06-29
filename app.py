import streamlit as st
from PIL import Image
import io
import sys
from pathlib import Path

# Add src folder to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from interface import predict_image

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Cloud Removal using U-Net",
    page_icon="🛰️",
    layout="wide"
)

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("🛰️ Project Details")

    st.markdown("---")

    st.write("### Model")
    st.write("U-Net")

    st.write("### Dataset")
    st.write("RICE1")

    st.write("### Framework")
    st.write("PyTorch")

    st.write("### Image Size")
    st.write("256 × 256")

    st.markdown("---")

    st.write("### Evaluation Metrics")

    st.metric("PSNR", "26.46 dB")
    st.metric("SSIM", "0.9001")

    st.markdown("---")

    st.success("✅ Model Loaded")

# =====================================================
# Title
# =====================================================

st.title("🛰️ Cloud Removal from Satellite Images")

st.markdown("""
### Bharatiya Antariksh Hackathon 2026

This application removes clouds from satellite images using a **U-Net Deep Learning Model**
trained on the **RICE (Remote Sensing Images for Cloud Removal)** dataset.

### Features

- Upload a cloudy satellite image
- Generate a cloud-free prediction
- Download the predicted image
- Powered by PyTorch & Streamlit
""")

st.info(
    "⚠️ Best results are obtained with images similar to the RICE dataset used during training."
)

# =====================================================
# Upload Image
# =====================================================

uploaded_file = st.file_uploader(
    "📤 Upload a Cloudy Satellite Image",
    type=["png", "jpg", "jpeg"]
)

# =====================================================
# Prediction
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Generating Cloud-Free Prediction..."):

        prediction = predict_image(image)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("☁️ Cloudy Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("🌍 Predicted Cloud-Free Image")

        st.image(
            prediction,
            use_container_width=True
        )

    # Download Button

    buffer = io.BytesIO()

    prediction.save(buffer, format="PNG")

    st.download_button(
        label="📥 Download Prediction",
        data=buffer.getvalue(),
        file_name="prediction.png",
        mime="image/png"
    )

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.markdown("""
## About the Project

This project was developed for the **Bharatiya Antariksh Hackathon 2026**.

### Technologies Used

- PyTorch
- U-Net Architecture
- Streamlit
- OpenCV
- PIL
- RICE Dataset

### Objective

To reconstruct cloud-free satellite images from cloudy remote sensing images using Deep Learning.
""")