#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tell the C++ compiler these exist somewhere else
extern const uint8_t dscnn_wake_model[];
extern const unsigned int dscnn_wake_model_len;

#ifdef __cplusplus
}
#endif

#endif // MODEL_H