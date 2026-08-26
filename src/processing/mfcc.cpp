#include "includes/mfcc.h"

extern const float32_t HANN_WINDOW[N_FFT];
extern const float32_t MEL_FILTERBANK[N_MELS * N_FFT_HALF]; 
extern const float32_t DCT_MATRIX[NUM_MFCC * N_MELS];

#define PAD_LEFT 16  // (512 - 480) / 2

MFCCExtractor::MFCCExtractor() {
    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);

    arm_mat_init_f32(&mel_matrix, N_MELS, N_FFT_HALF, (float32_t*)MEL_FILTERBANK);
    arm_mat_init_f32(&dct_matrix, NUM_MFCC, N_MELS, (float32_t*)DCT_MATRIX);

    arm_mat_init_f32(&power_spec_matrix, N_FFT_HALF, 1, power_spectrum);
    arm_mat_init_f32(&mel_energies_matrix, N_MELS, 1, mel_energies);
}

void MFCCExtractor::process_frame(const float32_t* audio_in, float32_t* mfcc_out) {
    // 1. Center the 480-sample windowed audio inside the 512 buffer
    // Zero out left pad
    for (int i = 0; i < PAD_LEFT; i++) {
        windowed_input[i] = 0.0f;
    }

    // Apply Hann window and copy into center (indices 16 to 495)
    arm_mult_f32(audio_in + PAD_LEFT, HANN_WINDOW, &windowed_input[PAD_LEFT], N_FFT);

    // Zero out right pad (indices 496 to 511)
    for (int i = PAD_LEFT + N_FFT; i < FFT_SIZE; i++) {
        windowed_input[i] = 0.0f;
    }

    // 2. 512-point Real FFT
    arm_rfft_fast_f32(&fft_instance, windowed_input, fft_output, 0);

    // 3. Unpack CMSIS-DSP packed format into power spectrum (257 bins)
    power_spectrum[0] = fft_output[0] * fft_output[0];

    for (int k = 1; k < N_FFT_HALF - 1; k++) {
        float32_t r = fft_output[2 * k];
        float32_t im = fft_output[2 * k + 1];
        power_spectrum[k] = (r * r) + (im * im);
    }

    power_spectrum[N_FFT_HALF - 1] = fft_output[1] * fft_output[1];

    // 4. Apply Mel Filterbank: (40 x 257) * (257 x 1) -> (40 x 1)
    arm_mat_mult_f32(&mel_matrix, &power_spec_matrix, &mel_energies_matrix);

    // 5. Decibel Scaling (10 * log10(x))
    for (int i = 0; i < N_MELS; i++) {
        if (mel_energies[i] < 1e-10f) {
            mel_energies[i] = 1e-10f;
        }
    }
    arm_vlog_f32(mel_energies, mel_energies, N_MELS);
    arm_scale_f32(mel_energies, 4.342944819f, mel_energies, N_MELS);

    // 6. DCT: (10 x 40) * (40 x 1) -> (10 x 1)
    arm_mat_init_f32(&mfcc_out_matrix, NUM_MFCC, 1, mfcc_out);
    arm_mat_mult_f32(&dct_matrix, &mel_energies_matrix, &mfcc_out_matrix);
}

// --- C Wrapper API Implementations ---
extern "C" {

mfcc_extractor_handle_t mfcc_create(void) {
    return reinterpret_cast<mfcc_extractor_handle_t>(new MFCCExtractor());
}

void mfcc_destroy(mfcc_extractor_handle_t handle) {
    if (handle) {
        delete reinterpret_cast<MFCCExtractor*>(handle);
    }
}

void mfcc_process_frame(mfcc_extractor_handle_t handle, const float* audio_in, float* mfcc_out) {
    if (handle && audio_in && mfcc_out) {
        reinterpret_cast<MFCCExtractor*>(handle)->process_frame(audio_in, mfcc_out);
    }
}

void mfcc_process_clip(mfcc_extractor_handle_t handle, const float* audio_16k, float* mfcc_10x49_out) {
    if (!handle || !audio_16k || !mfcc_10x49_out) return;

    MFCCExtractor* extractor = reinterpret_cast<MFCCExtractor*>(handle);
    const size_t num_frames = (16000 - FFT_SIZE) / HOP_LENGTH + 1; // 49 frames
    float temp_frame[NUM_MFCC];

    for (size_t t = 0; t < num_frames; ++t) {
        const float* frame_start = audio_16k + (t * HOP_LENGTH);
        extractor->process_frame(frame_start, temp_frame);

        for (size_t c = 0; c < NUM_MFCC; ++c) {
            mfcc_10x49_out[c * num_frames + t] = temp_frame[c];
        }
    }
}

} // extern "C"