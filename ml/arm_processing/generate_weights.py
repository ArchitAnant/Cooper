import torch
from processing import AudioProcessor

proc = AudioProcessor(sample_rate=16000)
dct_mat = proc.mfcc_transform.dct_mat.numpy().T.flatten()
mel_fb = proc.mfcc_transform.MelSpectrogram.mel_scale.fb.numpy().T.flatten()
hann_window = torch.hann_window(480, periodic=True).numpy()

with open("weights.cpp", "w") as f:
    f.write('#include "arm_math.h"\n\n')
    f.write('extern const float32_t HANN_WINDOW[480] = {\n' + ', '.join(f"{x:.8f}f" for x in hann_window) + '\n};\n\n')
    f.write('extern const float32_t MEL_FILTERBANK[40 * 257] = {\n' + ', '.join(f"{x:.8f}f" for x in mel_fb) + '\n};\n\n')
    f.write('extern const float32_t DCT_MATRIX[10 * 40] = {\n' + ', '.join(f"{x:.8f}f" for x in dct_mat) + '\n};\n\n')