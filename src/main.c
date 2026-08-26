#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include "processing/includes/clip.h"
#include "processing/includes/mfcc.h"
#include "test_audio.h"  // <-- Include your exported audio

int init_runtime(void);
int run_inference(float *input_features);

static float mfcc_features[NUM_MFCC * 49];

int main(void) {
    printk("[+] Boot Up\n");

    if (init_runtime()) return 0;
    
    mfcc_extractor_handle_t mfcc_handle = mfcc_create();
    
    // Use the real audio array instead of an empty buffer!
    clip_audio(test_audio_buffer);
    mfcc_process_clip(mfcc_handle, test_audio_buffer, mfcc_features);

    run_inference(mfcc_features);

    mfcc_destroy(mfcc_handle);
    printk("[+] Exiting...\n");
    return 0;
}