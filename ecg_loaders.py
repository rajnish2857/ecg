# ecg_loaders.py
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def image_to_1d_signal(uploaded_img) -> np.ndarray:
    """Converts any uploaded or captured ECG image into a normalized 1D waveform array."""
    pil_img = Image.open(uploaded_img).convert("RGB")
    rgb = np.array(pil_img, dtype=np.uint8)
    
    # Detect if image is a plotted chart with a colored (blue) waveform line
    r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    blue_mask = ((b - r) > 30) & ((b - g) > 15)
    
    if np.count_nonzero(blue_mask) > (rgb.shape * 0.5):
        # Extract strictly the blue waveform trace, ignoring black borders/axes
        binary = (blue_mask * 255).astype(np.uint8)
        # Trim empty left/right margins around the plot area
        cols_with_trace = np.where(np.any(binary > 0, axis=0))
        if len(cols_with_trace) > 10:
            binary = binary[:, cols_with_trace:cols_with_trace[-1] + 1]
    else:
        # Fallback for black-and-white ECG strips or camera photos
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    signal = []
    height, width = binary.shape
    for x in range(width):
        y_indices = np.where(binary[:, x] > 0)
        if len(y_indices) > 0:
            signal.append(height - np.mean(y_indices))
        else:
            signal.append(signal[-1] if signal else height / 2.0)
            
    sig_arr = np.array(signal, dtype=np.float32)
    return (sig_arr - np.mean(sig_arr)) / (np.std(sig_arr) + 1e-8)

def load_mitbih_csv(csv_file) -> pd.DataFrame:
    """Loads MIT-BIH CSV data and strips single/double quotes from column headers."""
    df = pd.read_csv(csv_file)
    df.columns = [col.strip().strip("'\"") for col in df.columns]
    return df

def extract_lead_from_df(df: pd.DataFrame, lead_col: str, normalize: bool = False) -> np.ndarray:
    """Extracts a clean 1D signal array from the chosen lead column (e.g., MLII or V5)."""
    raw_sig = pd.to_numeric(df[lead_col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if normalize:
        return (raw_sig - np.mean(raw_sig)) / (np.std(raw_sig) + 1e-8)
    return raw_sig

if __name__ == "__main__":
    print("ecg_loader running");