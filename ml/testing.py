from pathlib import Path

import torch
from model import MicroSpeechDSCNN, NUM_CLASSES, NUM_MFCC
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import random
import torchaudio.transforms as T
import torch.nn.functional as F
import torchaudio

MODEL_PATH = "models/best_kws_dscnn.pth"
PTE_MODEL_PATH = "models/kws_dscnn_portable.pte"
SAMPLE_RATE = 16000
CLIP_DURATION_SAMPLES = 16000  # 1 second
TESTING_DATA_DIR = "testing_audio"
TARGET_WORDS = ['yes', 'no', 'zero', 'one', 'two', 
                'three', 'left', 'right', 'up', 'down']
# We append 'unknown' and 'silence' to the end
CLASSES = TARGET_WORDS + ['unknown', 'silence']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

model = MicroSpeechDSCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, 
                                 map_location=torch.device('cpu')))
model.eval()

edge_runner = _load_for_executorch(PTE_MODEL_PATH)

print("[+] models loaded")

class KeywordSpottingDataset(Dataset):
    def __init__(self, file_paths, labels, bg_noises, is_training=True):
        self.file_paths = file_paths
        self.labels = labels
        self.bg_noises = bg_noises
        self.is_training = is_training
        
        self.mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            melkwargs={"n_fft": 480, "hop_length": 320, "n_mels": 40, "center": False}
        )

    def __len__(self):
        return len(self.file_paths)

    def _get_random_noise_slice(self):
        """Picks a random noise file and extracts a random 1-second window."""
        if not self.bg_noises:
            return torch.zeros((1, CLIP_DURATION_SAMPLES))
            
        noise_tensor = random.choice(self.bg_noises)
        max_start = noise_tensor.shape[1] - CLIP_DURATION_SAMPLES
        
        if max_start <= 0:
            return F.pad(noise_tensor, (0, abs(max_start))) # Pad if noise file is weirdly short
            
        start_idx = random.randint(0, max_start)
        return noise_tensor[:, start_idx : start_idx + CLIP_DURATION_SAMPLES]

    def _pad_or_trim(self, waveform):
        seq_len = waveform.shape[1]
        if seq_len < CLIP_DURATION_SAMPLES:
            pad_amount = CLIP_DURATION_SAMPLES - seq_len
            waveform = F.pad(waveform, (0, pad_amount))
        elif seq_len > CLIP_DURATION_SAMPLES:
            waveform = waveform[:, :CLIP_DURATION_SAMPLES]
        return waveform

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # --- Handle the 'Silence' class ---
        if file_path == "SILENCE_DUMMY_PATH":
            # For silence, just get a random slice of the background noise
            waveform = self._get_random_noise_slice()
            # Randomly vary the volume of the silence
            waveform = waveform * random.uniform(0.1, 0.8)
            
        # --- Handle Spoken Words ('yes', 'no', 'unknown') ---
        else:
            waveform, sr = torchaudio.load(file_path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            waveform = self._pad_or_trim(waveform)

            # Apply training augmentations
            if self.is_training:
                # 1. Random Time Shifting (As specified in the proposal)
                shift_amount = random.randint(-4000, 4000)
                waveform = torch.roll(waveform, shifts=shift_amount, dims=1)
                
                # 2. Add Background Noise (Mix real noise into the spoken word)
                # 80% chance to add noise
                if random.random() < 0.8:
                    noise = self._get_random_noise_slice()
                    # Keep noise volume relatively low compared to speech (0.0 to 0.2)
                    noise_volume = random.uniform(0.0, 0.2) 
                    waveform = waveform + (noise * noise_volume)
                    
            waveform = torch.clamp(waveform, -1.0, 1.0)

        # Generate MFCC KWS features (Expected shape: [1, 10, 49])
        mfcc = self.mfcc_transform(waveform)
        
        return mfcc, torch.tensor(label, dtype=torch.long)

def prepare_testing_data():
    """
    Prepares the testing data by creating a DataLoader for the test dataset.
    
    Args:
	test_dataset (Dataset): The test dataset to be loaded.
    
    Returns:
    """
    test_dataset = KeywordSpottingDataset(
	file_paths=[str(p) for p in Path(TESTING_DATA_DIR).rglob("*.wav")],
	labels=[CLASS_TO_IDX[p.parent.name] for p in Path(TESTING_DATA_DIR).rglob("*.wav")],
	bg_noises=[torchaudio.load(str(p))[0] for p in Path(TESTING_DATA_DIR).rglob("*.wav") if p.parent.name ==
 "background_noise"],
	is_training=False
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

err = []
acc = 0
test_loader = prepare_testing_data()
print("[+] Data loaded")

for i, (mfcc, label) in enumerate(test_loader):
    with torch.inference_mode():
        pytorch_output = model(mfcc)
        acc += (torch.argmax(pytorch_output, dim=1) == label).sum().item()
        # Ensure input tensor is passed properly as a sequence
        edge_outputs = edge_runner.forward((mfcc,))
        edge_output = edge_outputs[0]

    max_diff = torch.max(torch.abs(pytorch_output - edge_output)).item()
    err.append(max_diff)
    
    if max_diff >= 1e-5:
        print(f"[-]Verification warning for sample {i}! Diff: {max_diff:.8f}")

if err:
    print(f"\nVerification Complete Across {len(err)} Samples")
    print(f"  - Max Error: {max(err):.8f}")
    print(f"  - Avg Error: {sum(err)/len(err):.8f}")
    print(f"  - Accuracy: {acc}/{len(test_loader)} = {acc/len(test_loader)*100:.2f}%")