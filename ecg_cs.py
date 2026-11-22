import numpy as np
import pywt
from scipy.linalg import hadamard
import matplotlib.pyplot as plt

def get_sparse_dwt(signal: np.ndarray, wavelet: str = "db4", level: int = 4, thresh_ratio: float = 0.10):
    sig_centered = signal - np.mean(signal)
    coeffs = pywt.wavedec(sig_centered, wavelet=wavelet, level=level, mode="periodic")
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    thresh_val = thresh_ratio * np.max(np.abs(coeff_arr))
    sparse_arr = pywt.threshold(coeff_arr, value=thresh_val, mode="hard")
    return coeff_arr, sparse_arr

def generate_sensing_matrix(matrix_type: str, M: int, N: int, seed: int = 42) -> np.ndarray:
    M = int(M)
    N = int(1 << (int(N) - 1).bit_length())
    rng = np.random.default_rng(seed)
    
    if matrix_type == "Gaussian":
        return rng.standard_normal((M, N)) / np.sqrt(M)
    elif matrix_type == "Bernoulli":
        return rng.choice([-1.0, 1.0], size=(M, N)) / np.sqrt(M)
    elif matrix_type == "Hadamard":
        H = hadamard(N).astype(np.float64)
        row_indices = rng.permutation(N)[:M]
        return H[row_indices, :] / np.sqrt(M)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")

def compress_with_matrices(sparse_arr: np.ndarray, cr: float = 2.5) -> dict:
    orig_len = int(len(sparse_arr))
    N = int(1 << (orig_len - 1).bit_length())
    
    if N != orig_len:
        sparse_padded = np.pad(sparse_arr, (0, N - orig_len), mode="constant")
    else:
        sparse_padded = sparse_arr

    M = max(1, int(round(orig_len / cr)))
    
    compressed_outputs = {}
    for m_type in ["Gaussian", "Bernoulli", "Hadamard"]:
        phi = generate_sensing_matrix(m_type, M, N)
        y_compressed = phi @ sparse_padded
        compressed_outputs[m_type] = {
            "y": y_compressed,
            "M": M,
            "N": orig_len,
            "CR": orig_len / M
        }
    return compressed_outputs

def generate_all_plots(signal: np.ndarray, coeff_arr: np.ndarray, sparse_arr: np.ndarray,
                       compressed_outputs: dict, title_label: str):
    fig1, ax1 = plt.subplots(figsize=(12, 2.6))
    ax1.plot(signal, linewidth=0.9, color="#1f77b4")
    ax1.set_title(f"ECG Signal — {title_label}")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.6)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 2.6))
    ax2.plot(coeff_arr, linewidth=1.0, color="#1f77b4")
    ax2.set_title("DWT Wavelet Coefficients")
    ax2.set_xlabel("Coefficient Index")
    ax2.set_ylabel("Coefficient Value")
    ax2.grid(True, alpha=0.6)
    fig2.tight_layout()

    fig3, (ax3_top, ax3_bot) = plt.subplots(2, 1, figsize=(12, 4.2), sharex=True)
    ax3_top.plot(coeff_arr, linewidth=1.0, color="#1f77b4")
    ax3_top.set_title("Original DWT Coefficients")
    ax3_top.grid(True, alpha=0.6)
    
    sparse_plot = np.where(sparse_arr != 0, sparse_arr, np.nan)
    ax3_bot.plot(sparse_plot, linewidth=1.0, color="#1f77b4")
    ax3_bot.set_title("Sparse DWT Coefficients (Threshold = 0.10)")
    ax3_bot.set_xlabel("Coefficient Index")
    ax3_bot.set_ylim(ax3_top.get_ylim())
    ax3_bot.grid(True, alpha=0.6)
    fig3.tight_layout()

    fig4, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
    colors = {"Gaussian": "#2ca02c", "Bernoulli": "#ff7f0e", "Hadamard": "#9467bd"}
    
    for ax, (m_type, data) in zip(axes, compressed_outputs.items()):
        ax.plot(data["y"], linewidth=1.0, color=colors[m_type])
        ax.set_title(f"{m_type} Compressed Signal (y) — Length M = {data['M']} (from N = {data['N']}, CR = {data['CR']:.2f}x)")
        ax.set_ylabel("Compressed Value")
        ax.grid(True, alpha=0.6)
        
    axes[-1].set_xlabel("Compressed Measurement Index (0 to M-1)")
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4
