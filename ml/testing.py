from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ml.model import MicroSpeechDSCNN, NUM_CLASSES, NUM_MFCC
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from sklearn.metrics import classification_report

# Resolve paths relative to ml/ folder
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = SCRIPT_DIR / "models" / "best_kws_dscnn_32.pth"
PTE_MODEL_PATH = SCRIPT_DIR / "models" / "kws_dscnn_portable_32.pte"
SAMPLE_RATE = 16000
CLIP_DURATION_SAMPLES = 16000  # 1 second

# Check ml/testing_audio first, fallback to root testing_audio
if (SCRIPT_DIR / "testing_audio").exists():
    TESTING_DATA_DIR = SCRIPT_DIR / "testing_audio"
else:
    TESTING_DATA_DIR = Path("testing_audio")

TARGET_WORDS = ['yes', 'no', 'zero', 'one', 'two', 
                'three', 'left', 'right', 'up', 'down']
CLASSES = TARGET_WORDS + ['unknown', 'silence']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Load Models
model = MicroSpeechDSCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=torch.device('cpu')))
model.eval()
edge_runner = _load_for_executorch(str(PTE_MODEL_PATH))
print("[+] Models loaded")


class DomainTestDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            melkwargs={"n_fft": 480, "hop_length": 320, "n_mels": 40, "center": False}
        )

    def __len__(self):
        return len(self.file_paths)

    def _pad_or_trim(self, waveform):
        seq_len = waveform.shape[1]
        if seq_len < CLIP_DURATION_SAMPLES:
            return F.pad(waveform, (0, CLIP_DURATION_SAMPLES - seq_len))
        elif seq_len > CLIP_DURATION_SAMPLES:
            return waveform[:, :CLIP_DURATION_SAMPLES]
        return waveform

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = self._pad_or_trim(waveform)
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        mfcc = self.mfcc_transform(waveform)
        return mfcc, torch.tensor(self.labels[idx], dtype=torch.long), file_path


# Load domain test files
test_files, test_labels = [], []
for p in Path(TESTING_DATA_DIR).rglob("*.wav"):
    folder_name = p.parent.name
    if folder_name in CLASS_TO_IDX:
        test_files.append(str(p))
        test_labels.append(CLASS_TO_IDX[folder_name])

if len(test_files) == 0:
    print(f"[-] Error: No .wav files found in {TESTING_DATA_DIR.resolve()}")
    exit(1)

test_dataset = DomainTestDataset(test_files, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"[+] Loaded {len(test_dataset)} real domain test audio files from {TESTING_DATA_DIR}.")

all_preds = []
all_targets = []
mistakes = []

for mfcc, label, file_path in test_loader:
    with torch.inference_mode():
        pytorch_output = model(mfcc)
        pred_idx = torch.argmax(pytorch_output, dim=1).item()
        target_idx = label.item()

        all_preds.append(pred_idx)
        all_targets.append(target_idx)

        if pred_idx != target_idx:
            mistakes.append((file_path[0], CLASSES[target_idx], CLASSES[pred_idx]))

# Report
correct = sum(p == t for p, t in zip(all_preds, all_targets))
print("\n" + "=" * 50)
print(f"Domain Test Accuracy: {correct}/{len(all_targets)} = {correct/len(all_targets)*100:.2f}%")
print("=" * 50)

print("\nClassification Report:")
present_labels = sorted(list(set(all_targets)))
target_names = [CLASSES[i] for i in present_labels]
print(classification_report(all_targets, all_preds, labels=present_labels, target_names=target_names, zero_division=0))

if mistakes:
    print("\nMistakes Breakdown:")
    for path, truth, pred in mistakes:
        print(f" - File: {Path(path).name} | Ground Truth: '{truth}' -> Predicted: '{pred}'")