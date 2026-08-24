#pragma once
#include "arm_math.h"
#include <stdint.h>

#define N_FFT 480
#define FFT_SIZE 512
#define N_FFT_HALF 257  // (512 / 2) + 1
#define N_MELS 40
#define NUM_MFCC 10

class MFCCExtractor {
private:
    arm_rfft_fast_instance_f32 fft_instance;
    arm_matrix_instance_f32 mel_matrix;
    arm_matrix_instance_f32 dct_matrix;
    
    arm_matrix_instance_f32 power_spec_matrix;
    arm_matrix_instance_f32 mel_energies_matrix;
    arm_matrix_instance_f32 mfcc_out_matrix;

    float32_t windowed_input[FFT_SIZE];
    float32_t fft_output[FFT_SIZE];
    float32_t power_spectrum[N_FFT_HALF];
    float32_t mel_energies[N_MELS];

public:
    MFCCExtractor();
    void process_frame(const float32_t* audio_in, float32_t* mfcc_out);
};