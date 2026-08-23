# Cooper

Embedded Deep CNN (DCNN) keyword spotting pipeline running PyTorch models via the ExecuTorch runtime on Zephyr RTOS. Designed for bare-metal deployment on the Raspberry Pi Pico 2 with static memory planning (~126 KB RAM footprint).

---
Table of Contents
- [Directory Structure](#directory-structure)
- [Model Export Pipeline](#model-export-pipeline-ml)
- [Building & Flashing](#building--flashing)
- [Serial Console & Debugging](#serial-console--debugging)
- [MicroSpeechDSCNN: Ultra-Lightweight Keyword Spotting](#microspeechdscnn-ultra-lightweight-keyword-spotting)

---

## Directory Structure

```text
cooper/
├── CMakeLists.txt              # Application build rules and ExecuTorch linkage
├── prj.conf                    # Zephyr Kconfig profile
├── ml/                         # Training & AOT compilation pipeline
│   ├── execuwake_training.ipynb# Model training notebook
│   ├── model.py                # PyTorch DCNN architecture
│   ├── convert.py              # ExecuTorch .pte export & memory planning
│   ├── pte_to_array.py         # Flatbuffer byte array generator
│   └── models/                 # Checkpoints (.pth) and exported flatbuffers (.pte)
└── src/                        # Firmware execution engine
    ├── main.c                  # Application entry point
    └── dcnn/                   # Runtime wrapper & static memory pools

```

---

## Model Export Pipeline (`ml/`)

1. **Train Model:** Execute `execuwake_training.ipynb` to generate `models/best_kws_dscnn.pth`.
2. **Export to ExecuTorch:** Run `convert.py` to compile the PyTorch model to flatbuffer format (`kws_dscnn_portable.pte`).
3. **Generate C Header:** Run `pte_to_array.py` to embed the binary array directly into `src/dcnn/model.c`.

---

## Building & Flashing

Compile the firmware for the RP2350 Cortex-M33 target with CDC-ACM USB console support enabled:

```bash
west build -p always -b rpi_pico2/rp2350a/m33 -S cdc-acm-console .

```

To flash:

1. Hold **BOOTSEL** while plugging in the Pico 2 via USB.
2. Drag and drop `build/zephyr/zephyr.uf2` onto the mounted volume.

---

## Serial Console & Debugging

1. Identify the assigned serial port on macOS/Linux:
```bash
ls /dev/cu.usbmodem*

```


2. Connect to the terminal stream:
```bash
minicom -D /dev/cu.usbmodem101 -b 115200

```

---

## MicroSpeechDSCNN: Ultra-Lightweight Keyword Spotting

The `MicroSpeechDSCNN` is a highly constrained, memory-efficient acoustic model purpose-built for Always-On TinyML edge devices (e.g., microcontrollers, DSPs, and smart wearables). It processes 1-second audio clips via a lightweight 10-bin MFCC frontend (49 time frames) and is specifically optimized to execute within severe SRAM and Flash memory limits.

### Architectural Highlights

* **Depthwise Separable Convolutions:** Replaces standard, computationally heavy convolutions with DS-CNN blocks. By splitting the spatial (depthwise) and channel-mixing (pointwise) operations, it dramatically reduces both parameter count and multiply-accumulate (MAC) operations.
* **Temporal Shift Invariance:** Instead of flattening spatial dimensions into a massive, memory-heavy fully connected layer, the network utilizes **Global Average Pooling (GAP)**. This collapses the time and frequency domains, allowing the model to robustly recognize keywords regardless of exactly when they begin within the 1-second window.
* **Extreme Bottleneck Classifier:** The network methodically funnels feature maps down to exactly **8 channels** before the final classification head. For a standard multi-class setup, the final dense layer requires fewer than 100 weights, consuming virtually zero memory to execute the final decision.
* **Deployment Ready:** Integrates 2D Batch Normalization after every convolution to stabilize training. During Edge export, these layers can be mathematically folded/fused directly into the convolutional weights for zero-cost inference.

### Benchmark Performance

The model was evaluated on a strictly isolated test set, demonstrating excellent generalization with zero accuracy degradation between validation and unseen test data.

| Metric | Score |
| --- | --- |
| **Best Validation Accuracy** | 92.11% |
| **Unseen Test Accuracy** | 92.11% |
| **Final Test Loss** | 0.2372 |