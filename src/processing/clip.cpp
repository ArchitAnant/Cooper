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
#include <zephyr/sys/printk.h>

#define AUDIO_BUFFER_SIZE 16000 // 1s audio

float clip_num(float val){
    return std::max(-1.0f, std::min(1.0f, val));
}

void clip_audio(float *input_audio){
    if(input_audio == NULL){
        printk("[processsing][-] Input Audio is empty\n");
        return;
    }

    for(int i = 0; i < AUDIO_BUFFER_SIZE; i++){
        input_audio[i] = clip_num(input_audio[i]);
    }
    return;
}