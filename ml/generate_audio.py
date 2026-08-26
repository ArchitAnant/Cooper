import torch
import torchaudio

# Load a test audio file
waveform, sr = torchaudio.load("ml/testing_audio/yes/1.wav")

if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# Downmix to mono if stereo
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Pad or trim to exactly 16000 samples
if waveform.shape[1] < 16000:
    waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
elif waveform.shape[1] > 16000:
    waveform = waveform[:, :16000]

# Convert to a flat 1D numpy array
audio_np = waveform.detach().cpu().squeeze().numpy().flatten()

with open("test_audio.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("static float test_audio_buffer[16000] = {\n")
    f.write(", ".join([f"{float(x):.6f}f" for x in audio_np]))
    f.write("\n};\n")

print("Successfully exported test_audio.h!")