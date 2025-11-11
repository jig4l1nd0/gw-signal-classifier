import numpy as np
import os
import argparse
from tqdm import tqdm


def generate_chirp(length=2048, f_start=0.01, f_end=0.2):
    """Generates a sine wave with linearly increasing frequency (a chirp)."""
    t = np.arange(length)
    # Instantaneous frequency: f(t) = f_start + (f_end - f_start) * t / length
    # Phase (integral of frequency): phi(t)
    phi = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * length))

    # Apply envelope (e.g., Hann window) to make it start and end at zero
    envelope = np.hanning(length)

    signal = np.sin(phi) * envelope
    # Normalize to have a peak amplitude of 1
    return signal / np.max(np.abs(signal))


def generate_sample(length=2048, signal_length_min=400, signal_length_max=1000, noise_level=1.5):
    """Generates one noisy signal sample and its corresponding mask."""

    # 1. Create the clean signal (chirp)
    # We randomize the length of the chirp
    signal_length = np.random.randint(signal_length_min, signal_length_max)
    chirp = generate_chirp(length=signal_length)

    # 2. Create the mask (all zeros to start)
    mask = np.zeros(length, dtype=np.float32)

    # 3. Insert the signal at a random position
    clean_signal = np.zeros(length, dtype=np.float32)
    start_idx = np.random.randint(0, length - signal_length)
    end_idx = start_idx + signal_length

    clean_signal[start_idx:end_idx] = chirp
    mask[start_idx:end_idx] = 1  # Mark the signal region with '1'

    # 4. Create Gaussian noise
    noise = np.random.normal(0, noise_level, length)

    # 5. Add signal to noise
    noisy_signal = clean_signal + noise

    # 6. Normalize the final noisy signal (important for model stability)
    # We normalize by the standard deviation of the noise
    noisy_signal = noisy_signal / np.std(noisy_signal)

    return noisy_signal.astype(np.float32), mask


def main(args):
    """Main function to generate and save data."""
    noisy_signals = []
    masks = []

    print(f"Generating {args.num_samples} samples...")
    # Use tqdm for a nice progress bar
    for _ in tqdm(range(args.num_samples)):
        signal, mask = generate_sample(length=2048, noise_level=args.noise_level)
        noisy_signals.append(signal)
        masks.append(mask)

    noisy_signals_np = np.array(noisy_signals)
    masks_np = np.array(masks)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save files
    signals_path = os.path.join(args.output_dir, "noisy_signals.npy")
    masks_path = os.path.join(args.output_dir, "masks.npy")

    np.save(signals_path, noisy_signals_np)
    np.save(masks_path, masks_np)

    print("\nData generation complete.")
    print(f"Noisy signals shape: {noisy_signals_np.shape}")
    print(f"Masks shape: {masks_np.shape}")
    print(f"Data saved to {signals_path} and {masks_path}")


if __name__ == "__main__":
    # This allows us to run the script from the command line
    parser = argparse.ArgumentParser(description="Generate synthetic GW data.")
    parser.add_argument("--num-samples",
                        type=int, default=10000,
                        help="Number of samples to generate.")
    parser.add_argument("--output-dir",
                        type=str, default="data",
                        help="Directory to save the .npy files.")
    parser.add_argument("--noise-level",
                        type=float,
                        default=1.5,
                        help="Standard deviation of Gaussian noise.")
    args = parser.parse_args()
    main(args)
