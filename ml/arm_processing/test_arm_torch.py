import os
import torch
import torchaudio
import numpy as np
import arm_processing_bind as arm_dsp
from processing import AudioProcessor

SAMPLE_RATE = 16000
CLIP_DURATION_SAMPLES = 16000
TEST_FILE = "synthetic_test_audio.wav"

def generate_test_audio():
    """Generates 1 second of random noise audio and saves it to a file."""
    np.random.seed(42)
    # Generate random floats between -1.0 and 1.0
    raw_audio_np = np.random.uniform(-1.0, 1.0, CLIP_DURATION_SAMPLES).astype(np.float32)
    
    # torchaudio expects shape [channels, time]
    tensor_audio = torch.from_numpy(raw_audio_np).unsqueeze(0)
    torchaudio.save(TEST_FILE, tensor_audio, SAMPLE_RATE)
    
    return raw_audio_np

def run_verification():
    print("[+] Generating synthetic test audio...")
    raw_audio_np = generate_test_audio()

    # 1. Run PyTorch Pipeline
    print("[+] Running PyTorch AudioProcessor...")
    processor = AudioProcessor(sample_rate=SAMPLE_RATE)
    torch_mfcc = processor.process_audio(TEST_FILE)
    torch_mfcc_np = torch_mfcc.numpy()

    # 2. Run C++ CMSIS-DSP Pipeline
    print("[+] Running ARM CMSIS-DSP Pipeline...")
    # Passing the exact 1D numpy array, utilizing the hop_length and n_fft defaults
    arm_mfcc = arm_dsp.compute_mfcc(raw_audio_np, hop_length=320, n_fft=512)

    # 3. Compare Results
    print("\n" + "="*40)
    print("--- Shape Verification ---")
    print(f"PyTorch MFCC Shape: {torch_mfcc_np.shape}")
    print(f"ARM DSP MFCC Shape: {arm_mfcc.shape}")
    
    if torch_mfcc_np.shape != arm_mfcc.shape:
        print("[-] ERROR: Shapes do not match! Cannot compute mathematical difference.")
        return

    print("\n--- Frame 0 Comparison (First 5 Coefficients) ---")
    print("PyTorch: ", torch_mfcc_np[:5, 0])
    print("ARM DSP: ", arm_mfcc[:5, 0])

    print("\n--- Mathematical Verification ---")
    # Compute the maximum absolute difference between the two matrices
    max_diff = np.max(np.abs(torch_mfcc_np - arm_mfcc))
    mean_diff = np.mean(np.abs(torch_mfcc_np - arm_mfcc))
    
    print(f"Max Absolute Difference : {max_diff:.8f}")
    print(f"Mean Absolute Difference: {mean_diff:.8f}")
    print("="*40)

    # Note: Because C++ uses fast-math optimizations (like CMSIS rfft_fast), 
    # a tiny floating-point drift (e.g., 1e-4) is completely normal and acceptable.
    if max_diff < 1e-4:
        print("\n[SUCCESS] Verification Passed! The Edge frontend is mathematically identical.")
    else:
        print("\n[WARNING] Significant difference detected! See notes below.")

if __name__ == "__main__":
    try:
        run_verification()
    finally:
        # Clean up the temporary file
        if os.path.exists(TEST_FILE):
            os.remove(TEST_FILE)