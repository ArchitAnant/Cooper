# Cooper

Embedded Deep CNN (DCNN) keyword spotting pipeline running PyTorch models via the ExecuTorch runtime on Zephyr RTOS. Designed for bare-metal deployment on the Raspberry Pi Pico 2 with static memory planning (~84 KB RAM footprint).

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