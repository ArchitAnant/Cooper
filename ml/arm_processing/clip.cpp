/*
input: we get a wav file
processing:
- clip the values [-1., 1.]
- convert the input MFCC which shoudl ouput an 'tensor" 10x49
*/
#include "includes/clip.h"
#include <cstdint>
#include <stdint.h>
#include <algorithm>
#include <arm_math.h>

#define AUDIO_BUFFER_SIZE 16000 // 1s audio

float16_t clip_num(float16_t val){
    return std::max(static_cast<float16_t>(-1.0), std::min(static_cast<float16_t>(1.0), val));
}

void clip_audio(float *input_audio){
    if(input_audio == NULL){
        // printk("[processsing][-] Input Audio is empty\n");
        return;
    }

    for(int i = 0; i < AUDIO_BUFFER_SIZE; i++){
        input_audio[i] = clip_num(input_audio[i]);
    }
    return;
}