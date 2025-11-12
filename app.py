import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import io
from src.model import UNet1D


# Constants
MODEL_PATH = "unet_gw_model.pth"
EXAMPLE_SIGNAL_PATH = "example_signal.npy"
SIGNAL_LENGTH = 2048


@st.cache_resource
def load_model(model_path):
    """
    Loads the trained U-Net model.
    Uses st.cache_resource so the model is only loaded once.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'.")
        st.error("Make sure you have trained the model (scripts/train.py) "
                 "and 'unet_gw_model.pth' is in the root directory.")
        return None

    model = UNet1D(n_channels=1, n_classes=1)

    # Load the model on CPU (important for deployment)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )
    model.eval()  # Put the model in evaluation mode
    return model


@st.cache_data
def run_inference(_model, _signal_tensor):
    """
    Runs model inference on the input signal.
    Uses st.cache_data to avoid repeating calculations if signal unchanged.
    """
    with torch.no_grad():
        logits = _model(_signal_tensor)

    # Apply sigmoid to get probabilities (0 to 1)
    probs = torch.sigmoid(logits)

    # Get binary mask (0 or 1) using a threshold of 0.5
    mask = (probs > 0.5).float()

    # Move to numpy for plotting
    return probs.squeeze().numpy(), mask.squeeze().numpy()


def plot_results(signal, probs, mask):
    """
    Generates a plot with 3 subplots:
    1. Noisy input signal
    2. Probability mask (model output)
    3. Filtered signal (input * binary mask)
    """
    signal = signal.squeeze()  # Remove extra dimensions

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_alpha(0.0)  # Transparent background for Streamlit
    plt.style.use('dark_background')

    # 1. Noisy Input Signal
    ax1.plot(signal, label="Input signal", color="#4A90E2", alpha=0.8)
    ax1.set_title("1. Noisy Input Signal", color="white")
    ax1.set_ylabel("Amplitude", color="white")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')

    # 2. Probability Mask (U-Net Output)
    ax2.plot(probs, label="Probability (U-Net Output)", color="#50E3C2")
    ax2.axhline(0.5, ls='--', color='red', label='Threshold (0.5)')
    ax2.set_title("2. Model Prediction (Probability Mask)", color="white")
    ax2.set_ylabel("Probability", color="white")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors='white')

    # 3. Filtered Signal (Result)
    filtered_signal = signal * mask
    ax3.plot(filtered_signal, label="Filtered signal (Input * Mask)",
             color="#F5A623")
    ax3.set_title("3. Result: Isolated Signal", color="white")
    ax3.set_xlabel("Time Sample", color="white")
    ax3.set_ylabel("Amplitude", color="white")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(colors='white')

    plt.tight_layout()
    return fig


# --- Streamlit App Configuration ---
st.set_page_config(page_title="GW Classifier", layout="wide")

# Load the model
model = load_model(MODEL_PATH)

st.title("🛰️ Gravitational Wave Signal Classifier")
st.write("This app uses a 1D U-Net (in PyTorch) to detect and isolate "
         "chirp signals (like gravitational waves) within noisy data.")

# Add detailed instructions about data format
with st.expander("📋 How to prepare your data"):
    st.markdown("""
    ### Data Format Requirements
    
    To use this classifier with your own data, prepare your signal as follows:
    
    **File Format:**
    - Save as `.npy` file using `numpy.save()`
    - Example: `np.save('my_signal.npy', signal_array)`
    
    **Signal Requirements:**
    - **Shape**: Exactly `(2048,)` - a 1D NumPy array
    - **Data type**: Float32 or Float64
    - **Sampling**: Assumed to be evenly spaced time samples
    - **Normalization**: Signal should be roughly normalized (amplitude ~0-10)
    
    **Example Python code to prepare data:**
    ```python
    import numpy as np
    
    # Your signal data (must be exactly 2048 samples)
    my_signal = np.array([...])  # shape: (2048,)
    
    # Optional: normalize your signal
    my_signal = my_signal / np.std(my_signal)
    
    # Save as .npy file
    np.save('my_gravitational_wave_signal.npy', my_signal)
    ```
    
    **Signal Content:**
    - Time series data representing potential gravitational wave signals
    - Can contain noise (the model is trained to handle noisy signals)
    - Should represent physical measurements or simulated data
    - Expected frequency content: ~10-500 Hz (model training range)
    """)

    st.markdown("""
    ### What the model detects
    - **Chirp signals**: Frequency sweeps characteristic of gravitational waves
    - **Binary mergers**: Signals from black hole or neutron star coalescence
    - **Inspiral patterns**: Gradually increasing frequency and amplitude
    """)

# --- Sidebar ---
st.sidebar.header("Input Options")

# Add information about file format requirements
st.sidebar.markdown("### File Format Requirements")
st.sidebar.info(
    f"**Required format:**\n"
    f"- File type: `.npy` (NumPy array)\n"
    f"- Shape: `({SIGNAL_LENGTH},)` (1D array)\n"
    f"- Data type: Float32/Float64\n"
    f"- Content: Time series signal data\n"
    f"- Size: ~{SIGNAL_LENGTH * 4 / 1024:.1f} KB (float32)"
)

use_example = st.sidebar.button("Use Example Signal")

uploaded_file = st.sidebar.file_uploader(
    f"Upload your own signal (.npy, length {SIGNAL_LENGTH})",
    type=["npy"],
    help=f"Upload a NumPy array file (.npy) containing a 1D signal of exactly "
         f"{SIGNAL_LENGTH} samples. The signal should be normalized and "
         f"represent time series data similar to gravitational wave signals."
)

# --- Main Logic ---
signal_to_process = None

if use_example:
    if not os.path.exists(EXAMPLE_SIGNAL_PATH):
        st.error(f"'{EXAMPLE_SIGNAL_PATH}' not found. Please run first: "
                 "'python scripts/data_generator.py --generate-single "
                 "example_signal.npy'")
        st.stop()
    signal_to_process = np.load(EXAMPLE_SIGNAL_PATH)
    st.sidebar.success("Example signal loaded!")

elif uploaded_file is not None:
    # Load the .npy file from the uploader
    bytes_data = uploaded_file.getvalue()
    try:
        signal_to_process = np.load(io.BytesIO(bytes_data))
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading .npy file: {e}")
        st.stop()
else:
    st.info("← Use the sidebar to load a .npy signal or use the example.")
    st.stop()

# --- Processing and Visualization ---
if model and signal_to_process is not None:

    # 1. Validate the signal
    if signal_to_process.shape != (SIGNAL_LENGTH,):
        st.error(
            f"**Invalid signal shape!** 📐\n\n"
            f"**Expected:** `({SIGNAL_LENGTH},)` "
            f"(1D array with {SIGNAL_LENGTH} samples)\n"
            f"**Got:** `{signal_to_process.shape}`\n\n"
            f"**How to fix:**\n"
            f"- Ensure your signal is a 1D NumPy array\n"
            f"- Reshape if needed: `signal = signal.reshape(-1)` "
            f"or `signal.flatten()`\n"
            f"- Resample to {SIGNAL_LENGTH} samples if your signal is "
            f"longer/shorter\n"
            f"- Check the expandable section above for data preparation "
            f"examples"
        )
        st.stop()

    # 2. Preprocess the signal for the model
    # (Length,) -> (Batch, Channels, Length) -> (1, 1, 2048)
    signal_tensor = torch.from_numpy(signal_to_process).float().unsqueeze(
        0).unsqueeze(0)

    # 3. Run inference
    with st.spinner("Running model..."):
        probs, mask = run_inference(model, signal_tensor)

    # 4. Show results
    st.header("Detection Results")
    fig = plot_results(signal_to_process, probs, mask)
    st.pyplot(fig)
