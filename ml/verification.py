import torch
from ml.model import MicroSpeechDSCNN, NUM_CLASSES, NUM_MFCC
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from executorch.exir._serialize import _deserialize_pte_binary

TIME_FRAMES = 49
MODEL_PATH = "ml/models/best_kws_dscnn_32_512.pth"
PTE_MODEL_PATH = "ml/models/kws_dscnn_portable_32_512.pte"

print("Analyzing ExecuTorch model memory requirements...")

try:
    with open(PTE_MODEL_PATH, "rb") as f:
        pte_data = f.read()
        
    # 1. Properly deserialize the raw .pte binary into the FlatBuffers wrapper
    exec_prog = _deserialize_pte_binary(pte_data)
    
    # 2. Access the internal execution plan list from the program structure
    # FlatBuffers array structures use explicit methods or list fields depending on the schema
    execution_plans = exec_prog.program.execution_plan
    
    if not execution_plans:
        print("Error: No execution plans found in this model file.")
    else:
        # 3. Read the AOT non-constant activation buffer pool requirements 
        buffer_sizes = execution_plans[0].non_const_buffer_sizes
        
        print("\n--- Edge Memory Requirements ---")
        if not buffer_sizes:
            print("Pool 0 (activation_pool): 0 bytes (No dynamic activations planned)")
        else:
            for i, size in enumerate(buffer_sizes):
                print(f"Pool {i} (activation_pool) requires EXACTLY: {size} bytes ({size / 1024:.2f} KB)")
                
except FileNotFoundError:
    print(f"Error: Could not find the file at {PTE_MODEL_PATH}")
except Exception as e:
    print(f"An unexpected error occurred during deserialization: {e}")

print("--------------------------------\n")


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