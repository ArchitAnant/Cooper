#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>
#include "arm_math.h"

#define N_FFT 480
#define FFT_SIZE 512
#define N_FFT_HALF 257  /* (512 / 2) + 1 */
#define N_MELS 40
#define NUM_MFCC 10
#define HOP_LENGTH 320

#ifdef __cplusplus
/* C++ Class Definition */
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

extern "C" {
#endif

/* --- C API Declarations --- */
typedef struct MFCCExtractorOpaque* mfcc_extractor_handle_t;

/* Lifecycle and processing handles for C code */
mfcc_extractor_handle_t mfcc_create(void);
void mfcc_destroy(mfcc_extractor_handle_t handle);
void mfcc_process_frame(mfcc_extractor_handle_t handle, const float* audio_in, float* mfcc_out);

/* Convenience function: Process an entire 1-second (16000 sample) buffer to (10 x 49) matrix */
void mfcc_process_clip(mfcc_extractor_handle_t handle, const float* audio_16k, float* mfcc_10x49_out);

#ifdef __cplusplus
}
#endif

#endif /* MFCC_H */