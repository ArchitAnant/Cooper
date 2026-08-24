/*
input: we get a wav file
processing:
- clip the values [-1., 1.]
- convert the input MFCC which shoudl ouput an 'tensor" 10x49
*/

#include <stdint.h>
#include <algorithm>
#include <zephyr/sys/printk.h>
#include <arm_math.h>

#define AUDIO_BUFFER_SIZE 16000 // 1s audio

float16_t clip(float16_t val){
    return std::max(-1.0, std::min(1.0, val));
}

int clip_audio(float16_t *input_audio){
    if(input_audio == NULL)
        printk("[processsing][-] Input Audio is empty\n");
        return -1;

    for(int i = 0; i < AUDIO_BUFFER_SIZE; i++){
        input_audio[i] = clip(input_audio[i]);
    }
    return 0;
}