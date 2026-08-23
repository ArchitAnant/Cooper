import torch
print(torch.__version__)
from torch.export import export
import torch._inductor.utils
if not hasattr(torch._inductor.utils, 'XPU_KERNEL_FORMAT'):
    torch._inductor.utils.XPU_KERNEL_FORMAT = None

from executorch.exir import to_edge
from ml.model import MicroSpeechDSCNN, NUM_CLASSES, NUM_MFCC

MODEL_PATH = "ml/models/best_kws_dscnn_32.pth"
PTE_MODEL_PATH = "ml/models/kws_dscnn_portable_32.pte"
TIME_FRAMES = 49 # From 16kHz, 1s duration, 30ms window, 20ms stride

model = MicroSpeechDSCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

model.eval()

example_input = (torch.randn(1, 1, NUM_MFCC, TIME_FRAMES),)

print("1. Exporting PyTorch model to ATen graph...")
exported_program = export(model, example_input)

print("2. Lowering to ExecuTorch Edge dialect...")
edge_program = to_edge(exported_program)

print("3. Generating final ExecuTorch binary...")
executorch_program = edge_program.to_executorch()

# --- 3. Save the Flatbuffer ---
output_filename = PTE_MODEL_PATH
with open(output_filename, "wb") as f:
    f.write(executorch_program.buffer)

print(f"\nSuccess! ExecuTorch model saved to: {output_filename}")