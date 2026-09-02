import argparse
import struct
import time

import numpy as np
import torch
import torchaudio
import serial


NUM_SAMPLES = 16000
BAUD_RATE = 921600
HANDSHAKE = 0xAA

DEFAULT_PORTS = [
    "/dev/cu.usbmodem1101",
    "/dev/cu.usbmodem101",
    "/dev/cu.usbmodem1011",
]

TARGET_WORDS = ['yes', 'no', 'zero', 'one', 'two', 
                'three', 'left', 'right', 'up', 'down']
# We append 'unknown' and 'silence' to the end
CLASSES = TARGET_WORDS + ['unknown', 'silence']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_audio(path):
    """
    Load WAV and convert it to exactly 16000 float32 samples.

    This matches the preprocessing used by the existing
    test_audio.h generation script.
    """

    waveform, sr = torchaudio.load(path)

    print(f"[+] Loaded audio: {path}")
    print(f"[+] Original sample rate: {sr}")
    print(f"[+] Original shape: {tuple(waveform.shape)}")

    # Resample to 16 kHz
    if sr != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            16000
        )

    # Downmix to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(
            waveform,
            dim=0,
            keepdim=True
        )

    # Pad or trim to exactly 16000 samples
    if waveform.shape[1] < NUM_SAMPLES:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, NUM_SAMPLES - waveform.shape[1])
        )
    elif waveform.shape[1] > NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]

    # Convert to flat float32 NumPy array
    audio_np = (
        waveform
        .detach()
        .cpu()
        .squeeze()
        .numpy()
        .flatten()
        .astype(np.float32)
    )

    assert audio_np.shape == (NUM_SAMPLES,)
    assert audio_np.dtype == np.float32

    print(f"[+] Final samples: {len(audio_np)}")
    print(f"[+] Final dtype: {audio_np.dtype}")
    print(f"[+] Payload size: {audio_np.nbytes} bytes")

    return audio_np


def connect(port, baud):
    print(f"[+] Opening {port} @ {baud} baud")

    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=5,
        write_timeout=5,
    )

    # Opening the CDC ACM device can reset the Pico.
    time.sleep(1)

    ser.reset_input_buffer()

    return ser


def wait_for_host_ack(ser, max_retries=10):
    """
    Send handshake byte and wait for the same byte back.
    Retries multiple times in case the Pico is booting or out of sync.
    """
    print("[+] Waiting for Pico handshake...")
    
    # Temporarily shorten timeout for faster retries
    original_timeout = ser.timeout
    ser.timeout = 1.0  

    for attempt in range(max_retries):
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        ser.write(bytes([HANDSHAKE]))
        ser.flush()

        response = ser.read(1)

        if response == bytes([HANDSHAKE]):
            print("[+] Connection established")
            ser.timeout = original_timeout # Restore original timeout
            return
            
        print(f"[-] Handshake attempt {attempt + 1}/{max_retries} failed (got {response!r}). Retrying...")
        time.sleep(0.5)

    ser.timeout = original_timeout
    raise RuntimeError("Handshake failed after multiple retries. Is the Pico running the right code?")


def send_audio(ser, audio):
    """
    Send exactly 16000 float32 values.

    16000 × 4 = 64000 bytes.
    """

    payload = audio.tobytes()

    if len(payload) != NUM_SAMPLES * 4:
        raise RuntimeError(
            f"Invalid payload size: {len(payload)}"
        )

    print(f"[+] Sending {len(payload)} bytes...")

    start = time.monotonic()

    ser.write(payload)
    ser.flush()

    elapsed = time.monotonic() - start

    print(f"[+] Audio sent in {elapsed:.3f} seconds")


def receive_result(ser):
    """
    Receive PredResult:

        int32  class_id
        float32 confidence

    Total: 8 bytes
    """

    print("[+] Waiting for prediction...")

    data = ser.read(8)

    if len(data) != 8:
        raise RuntimeError(
            f"Expected 8 result bytes, received {len(data)}"
        )

    class_id, confidence = struct.unpack("<if", data)

    return class_id, confidence


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "audio",
        help="Path to WAV file"
    )

    parser.add_argument(
        "--port",
        default=None,
        help="Pico serial port"
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"Baud rate (default: {BAUD_RATE})"
    )

    args = parser.parse_args()

    audio = load_audio(args.audio)

    if args.port is not None:
        port = args.port
    else:
        port = None

        for candidate in DEFAULT_PORTS:
            try:
                ser = connect(candidate, args.baud)
                port = candidate
                break
            except serial.SerialException:
                pass

        if port is None:
            raise RuntimeError(
                "Could not find Pico USB serial device"
            )
    try:
        if args.port is not None:
            ser = connect(args.port, args.baud)

        wait_for_host_ack(ser)

        send_audio(ser, audio)

        class_id, confidence = receive_result(ser)

        print()
        print("========== Prediction ==========")
        print(f"Class      : {CLASSES[class_id]}")
        print(f"Confidence : {confidence:.6f}")
        print("================================")

    finally:
        ser.close()


if __name__ == "__main__":
    main()