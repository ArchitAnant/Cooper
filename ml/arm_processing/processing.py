import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T

CLIP_DURATION_SAMPLES = 16000 * 1
SAMPLE_RATE = 16000
NUM_MFCC = 10

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            melkwargs={
                "n_fft": 512,          # Match CMSIS-DSP 512-point FFT
                "win_length": 480,     # 30ms window (480 samples)
                "hop_length": 320,     # 20ms stride (320 samples)
                "n_mels": 40,
                "center": False
            }
        )
        
    def _pad_or_trim(self, waveform):
        seq_len = waveform.shape[1]
        if seq_len < CLIP_DURATION_SAMPLES:
            return F.pad(waveform, (0, CLIP_DURATION_SAMPLES - seq_len))
        elif seq_len > CLIP_DURATION_SAMPLES:
            return waveform[:, :CLIP_DURATION_SAMPLES]
        return waveform

    def load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        return waveform

    def process_audio(self, audio_path):
        waveform = self.load_audio(audio_path)
        waveform = self._pad_or_trim(waveform)
        waveform = torch.clamp(waveform, -1.0, 1.0)

        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0)  # Remove channel dimension if present
        return mfcc