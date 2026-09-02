# Cooper

Embedded Deep CNN (DCNN) keyword spotting pipeline running PyTorch models via the ExecuTorch runtime on Zephyr RTOS. Designed for bare-metal deployment on the Raspberry Pi Pico 2 with static memory planning (~126 KB RAM footprint).

## Tabel of Contents
- [Directory Structure](#directory-structure)
- [Model Export Pipeline](#model-export-pipeline)
- [Host-Device Communication](#host-device-communication)
- [Building & Flashing](#building--flashing)
- [MicroSpeechDSCNN](#microspeechdscnn)
  - [Architectural Highlights](#architectural-highlights)
  - [Benchmark Performance](#benchmark-performance)

## Directory Structure

```text
cooper/
├── CMakeLists.txt              # Application build rules and ExecuTorch linkage
├── comms_testing.py            # High-speed UART test script for audio payloads
├── ml/                         # Training, processing bindings & AOT compilation
│   ├── arm_processing/         # C++ MFCC/Clip bindings for Python-to-C parity
│   ├── execuwake_training.ipynb# Model training notebook
│   ├── convert.py              # ExecuTorch .pte export & memory planning
│   ├── pte_to_array.py         # Flatbuffer byte array generator
│   └── models/                 # Checkpoints (.pth) and exported flatbuffers (.pte)
└── src/                        # Firmware execution engine
    ├── main.c                  # Application entry point
    ├── comms/                  # UART host-device synchronization API
    ├── processing/             # Embedded C++ audio processing (MFCC/Clip)
    └── dcnn/                   # Runtime wrapper & static memory pools

```

## Model Export Pipeline

* **Train Model:** Execute `execuwake_training.ipynb` to generate `.pth` checkpoints.


* **Audio Preprocessing:** The `ml/arm_processing` module compiles the embedded C++ MFCC extraction into Python via pybind11, guaranteeing bit-exact feature parity between training data and live device inference.
* **Export to ExecuTorch:** Run `convert.py` to compile the PyTorch model to flatbuffer format (`.pte`).


* **Generate C Header:** Run `pte_to_array.py` to embed the binary array directly into `src/dcnn/model.c`.



## Host-Device Communication

The integrated `comms_testing.py` acts as the bridge between your host machine and the Pico.

* Synchronizes with the device's `src/comms/uart_api.c` firmware via a blocking handshake byte (`0xAA`).
* Streams exactly 16,000 float32 audio samples (64kB) to the device over a 921,600 baud UART connection.
* Awaits the firmware's inference cycle and reads back the structured 8-byte prediction result (Class ID and Confidence).

## Building & Flashing

Compile the firmware for the RP2350 Cortex-M33 target with CDC-ACM USB console support enabled:

```bash
west build -p always -b rpi_pico2/rp2350a/m33 -S cdc-acm-console .

```

* Hold **BOOTSEL** while plugging in the Pico 2 via USB.


* Drag and drop `build/zephyr/zephyr.uf2` onto the mounted volume.



## MicroSpeechDSCNN

The `MicroSpeechDSCNN` is a highly constrained, memory-efficient acoustic model purpose-built for Always-On TinyML edge devices (e.g., microcontrollers, DSPs, and smart wearables). It processes 1-second audio clips via a lightweight 10-bin MFCC frontend (49 time frames) and is specifically optimized to execute within severe SRAM and Flash memory limits.

### Architectural Highlights

* **Depthwise Separable Convolutions:** Replaces standard, computationally heavy convolutions with DS-CNN blocks. By splitting the spatial (depthwise) and channel-mixing (pointwise) operations, it dramatically reduces both parameter count and multiply-accumulate (MAC) operations.


* **Temporal Shift Invariance:** Instead of flattening spatial dimensions into a massive, memory-heavy fully connected layer, the network utilizes Global Average Pooling (GAP). This collapses the time and frequency domains, allowing the model to robustly recognize keywords regardless of exactly when they begin within the 1-second window.


* **Extreme Bottleneck Classifier:** The network methodically funnels feature maps down to exactly 8 channels before the final classification head. For a standard multi-class setup, the final dense layer requires fewer than 100 weights, consuming virtually zero memory to execute the final decision.


* **Deployment Ready:** Integrates 2D Batch Normalization after every convolution to stabilize training. During Edge export, these layers can be mathematically folded/fused directly into the convolutional weights for zero-cost inference.



### Benchmark Performance

The model was evaluated on a strictly isolated test set, demonstrating excellent generalization with zero accuracy degradation between validation and unseen test data.

| Metric | Score |
| --- | --- |
| **Best Validation Accuracy** | 92.11%|
| **Unseen Test Accuracy** | 92.11%|
| **Final Test Loss** | 0.2372 |