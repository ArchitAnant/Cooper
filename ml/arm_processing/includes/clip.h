#pragma once

#include <cstdint>

#ifndef float16_t
typedef _Float16 float16_t;
#endif

#ifndef AUDIO_BUFFER_SIZE
#define AUDIO_BUFFER_SIZE 16000
#endif

void clip_audio(float *input_audio);