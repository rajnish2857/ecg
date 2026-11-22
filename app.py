# app.py
import streamlit as st
import numpy as np
from ecg_loaders import image_to_1d_signal, load_mitbih_csv, extract_lead_from_df
from ecg_cs import get_sparse_dwt, compress_with_matrices, generate_all_plots

st.set_page_config(page_title="ECG DWT + CS Compression", layout="wide")
st.title("ECG Compression: DWT Sparsification + Gaussian, Bernoulli & Hadamard")

# Sidebar controls
st.sidebar.header("Compression Settings")
wavelet = st.sidebar.selectbox("Wavelet Family", ["db4", "db6", "sym4"], index=0)
level = st.sidebar.slider("DWT Level", 1, 6, 4)
thresh_ratio = st.sidebar.number_input("Sparsity Threshold", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
target_cr = st.sidebar.slider("Compression Ratio (N / M)", min_value=1.5, max_value=8.0, value=2.5, step=0.5)

def resample_to_1024(sig: np.ndarray) -> np.ndarray:
    """Resamples or slices signal to exact N=1024 samples (required for Hadamard matrix)."""
    if len(sig) >= 1024:
        return sig[:1024]
    x_old = np.linspace(0, 1, len(sig))
    x_new = np.linspace(0, 1, 1024)
    return np.interp(x_new, x_old, sig)

tab_img, tab_cam, tab_csv = st.tabs([
    "📁 Direct Image Uploader",
    "📸 Photo Clicker",
    "📊 MIT-BIH CSV Data"
])

ecg_signal = None
title_label = ""

with tab_img:
    img_file = st.file_uploader(
        "Upload ECG Image (Any Format)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"]
    )
    if img_file is not None:
        st.image(img_file, caption=img_file.name, use_container_width=True)
        raw_sig = image_to_1d_signal(img_file) * 100.0 + 1000.0
        ecg_signal = resample_to_1024(raw_sig)
        title_label = f"Image ({img_file.name})"

with tab_cam:
    cam_photo = st.camera_input("Click ECG Photo")
    if cam_photo is not None:
        raw_sig = image_to_1d_signal(cam_photo) * 100.0 + 1000.0
        ecg_signal = resample_to_1024(raw_sig)
        title_label = "Camera Capture"

with tab_csv:
    csv_file = st.file_uploader("Upload MIT-BIH CSV", type=["csv"])
    if csv_file is not None:
        df = load_mitbih_csv(csv_file)
        lead_options = [c for c in df.columns if "sample" not in c.lower()]
        selected_lead = st.selectbox("Select ECG Lead:", lead_options)
        full_sig = extract_lead_from_df(df, selected_lead, normalize=False)
        ecg_signal = resample_to_1024(full_sig)
        title_label = f"Record 100 ({selected_lead})"

if ecg_signal is not None:
    st.divider()
    
    # 1. DWT + 0.10 Sparse Threshold
    coeff_arr, sparse_arr = get_sparse_dwt(
        ecg_signal, wavelet=wavelet, level=level, thresh_ratio=thresh_ratio
    )
    
    # 2. Compress with Gaussian, Bernoulli, and Hadamard matrices
    compressed_outputs = compress_with_matrices(sparse_arr, cr=target_cr)
    
    # Summary Metrics
    non_zeros = int(np.count_nonzero(sparse_arr))
    sparsity_pct = (1.0 - non_zeros / len(sparse_arr)) * 100.0
    m_len = compressed_outputs["Gaussian"]["M"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Length (N)", f"{len(ecg_signal)}")
    c2.metric("Non-Zero DWT Coeffs", f"{non_zeros} ({sparsity_pct:.1f}% sparse)")
    c3.metric("Compressed Length (M)", f"{m_len}")
    c4.metric("Compression Ratio", f"{len(ecg_signal)/m_len:.2f}x")
    
    # Render all 4 figures
    fig1, fig2, fig3, fig4 = generate_all_plots(
        ecg_signal, coeff_arr, sparse_arr, compressed_outputs, title_label
    )
    
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.subheader("Compressed Measurements (y = Φ · x_sparse)")
    st.pyplot(fig4)
