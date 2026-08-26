#ifndef CLIP_H
#define CLIP_H

#include <stdint.h>

#ifndef AUDIO_BUFFER_SIZE
#define AUDIO_BUFFER_SIZE 16000
#endif

#ifdef __cplusplus
extern "C" {
#endif

void clip_audio(float *input_audio);

#ifdef __cplusplus
}
#endif

#endif /* CLIP_H */