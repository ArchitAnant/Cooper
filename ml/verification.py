import torch
from ml.model import MicroSpeechDSCNN, NUM_CLASSES, NUM_MFCC
from executorch.extension.pybindings.portable_lib import _load_for_executorch

TIME_FRAMES = 49
MODEL_PATH = "model/best_kws_dscnn.pth"
PTE_MODEL_PATH = "model/kws_dscnn_portable.pte"

print("Loading ExecuTorch model for verification...")
edge_runner = _load_for_executorch(PTE_MODEL_PATH)

test_tensor = torch.randn(1, 1, NUM_MFCC, TIME_FRAMES)

model = MicroSpeechDSCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Run the original PyTorch model
with torch.inference_mode():
    pytorch_output = model(test_tensor)

# Run the ExecuTorch model
# .forward() returns a tuple, so we grab the first element
edge_output = edge_runner.forward((test_tensor,))[0]

print("\n--- Output Comparison ---")
print(f"PyTorch Output:    {pytorch_output.numpy()}")
print(f"ExecuTorch Output: {edge_output.numpy()}")

# Assert they are mathematically identical
max_diff = torch.max(torch.abs(pytorch_output - edge_output)).item()
print(f"Max difference between outputs: {max_diff:.8f}")

if max_diff < 1e-5:
    print("Verification Passed! The Edge model is mathematically identical.")