#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#ifdef __cplusplus
extern "C" {
    #endif

    int init_runtime();
    int run_inference(float* input_data);

    #ifdef __cplusplus
}
#endif
#endif