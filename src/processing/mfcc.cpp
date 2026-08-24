#include "arm_math.h"
#include <stdint.h>

#define N_FFT 480
#define N_FFT_HALF 241
#define N_MELS 40
#define NUM_MFCC 10

extern const float32_t HANN_WINDOW[N_FFT];
extern const float32_t MEL_FILTERBANK[N_MELS * N_FFT_HALF];
extern const float32_t DCT_MATRIX[NUM_MFCC * N_MELS];

class MFCCExtractor {
        private:
                arm_rftt_fast_instance_f32 fft_instance;
                arm_matrix_instance_f32 mel_matrix;
                arm_matrix_instance_f32 dct_matrix;

                arm_matrix_instance_f32 power_spec_matrix;
                arm_matrix_instance_f32 mel_energies_matrix;
                arm_matrix_instance_f32 mfcc_out_matrix;

                float32_t windowed_input[N_FFT];
                float32_t fft_output[N_FFT];
                float32_t power_spectrum[N_FFT_HALF];
                float32_t mel_energies[N_MELS];

        public:
                MFCCExtractor(){
                        arm_rfft_fast_init_f32(&fft_instance, 512);

                        arm_mat_init_f32(&mel_matrix, N_MELS, N_FFT_HALF, (float32_t*)MEL_FILTERBANK);
                        arm_mat_init_f32(&dct_matrix, NUM_MFCC, N_MELS, (float32_t*)DCT_MATRIX);

                        arm_mat_init_f32(&power_spec_matrix, N_FFT_HALF, 1, power_spectrum);
                        arm_mat_init_f32(&mel_energies_matrix, N_MELS, 1, mel_energies);
                }

                void process_frame(const float32_t* audio_in, float32_t* mfcc_out){
                        arm_mult_f32(audio_in, HANN_WINDOW, windowed_input, N_FFT);

                        for(int i = N_FFT; i<512; i++)
                                windowed_input[i] = 0.0f;
                        
                        arm_rfft_fast_f32(&fft_output, power_spectrum, fft_output, 0);
                        arm_cmplx_mag_squared_f32(fft_output, power_spectrum, N_FFT_HALF);

                        arm_mat_mult_f32(&mel_matrix, &power_spec_matrix, &mel_energies_matrix);

                        float32_t log_offset = 1e-6f;

                        for(int i = 0;i < N_MELS; i++)
                                mel_energies[i] += log_offset;

                        arm_vlog_f32(mel_energies, mel_energies, N_MELS);

                        arm_scale_f32(mel_energies, 4.342944819f, mel_energies, N_MELS);

                        arm_mat_init_f32(&mfcc_out_matrix, NUM_MFCC, 1, mfcc_out);
                        arm_mat_mult_f32(&dct_matrix, &mel_energies_matrix, &mfcc_out_matrix);
                }

}