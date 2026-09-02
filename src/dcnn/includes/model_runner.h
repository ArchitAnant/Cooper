#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H


#ifdef __cplusplus
extern "C" {
    #endif

    typedef struct{
        int class_id;
        float confidence;
    } PredResult;

    int init_runtime();
    PredResult run_inference(float* input_data);

    #ifdef __cplusplus
}
#endif
#endif