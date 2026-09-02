#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "includes/model.h"
#include "includes/const.hpp"
#include "includes/model_runner.h"
#include <cmath>
#include <memory>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

using namespace executorch::runtime;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using ScalarType = executorch::runtime::etensor::ScalarType;

// --- 1. Memory Pools (Moved from header) ---
static uint8_t method_allocator_pool[48 * 1024];
static uint8_t activation_pool[126 * 1024];

// --- 2. Global Pointers (Lifetimes fix) ---
std::unique_ptr<Program> program_ptr;
std::unique_ptr<Method> method_ptr;

std::unique_ptr<MemoryAllocator> method_allocator;
std::unique_ptr<HierarchicalAllocator> planned_memory;
std::unique_ptr<MemoryManager> memory_manager;

static Span<uint8_t> memory_planned_buffers[1];

int init_runtime() {
    runtime_init();

    // 3. Dynamically allocate the managers so they survive function exit
    method_allocator = std::make_unique<MemoryAllocator>(sizeof(method_allocator_pool), method_allocator_pool);
    
    memory_planned_buffers[0] = {activation_pool, sizeof(activation_pool)};
    planned_memory = std::make_unique<HierarchicalAllocator>(Span<Span<uint8_t>>(memory_planned_buffers, 1));
    
    memory_manager = std::make_unique<MemoryManager>(method_allocator.get(), planned_memory.get());

    // printk("[+] Loading model binary (%u bytes)...\n", dscnn_wake_model_len);
    executorch::extension::BufferDataLoader loader(dscnn_wake_model, dscnn_wake_model_len);

    auto program_result = Program::load(&loader);
    if (!program_result.ok()) {
        // printk("[-] Failed to load Program flatbuffer!\n");
        return -1;
    }
    program_ptr = std::make_unique<Program>(std::move(*program_result));
    // printk("[+] Program loaded successfully.\n");

    // Pass the global memory manager pointer
    auto method_result = program_ptr->load_method("forward", memory_manager.get());
    if (!method_result.ok()) {
        // printk("[-] Failed to load 'forward' method!\n");
        return -1;
    }
    method_ptr = std::make_unique<Method>(std::move(*method_result));
    // printk("[+] Method 'forward' instantiated in memory.\n");

    return 0;
}

PredResult run_inference(float* input_data) {

    TensorImpl::SizesType input_sizes[4] = {1, 1, NUM_MFCC, TIME_FRAMES};
    TensorImpl::DimOrderType dim_order[4] = {0, 1, 2, 3};

    TensorImpl input_impl(ScalarType::Float, 4, input_sizes, input_data, dim_order);
    Tensor input(&input_impl);

    PredResult result;

    if (method_ptr->set_input(input, 0) != Error::Ok) {
        // printk("[-] Failed to bind input tensor!\n");
        return result;
    }

    // printk("[+] Running inference pass...\n");
    if (method_ptr->execute() != Error::Ok) {
        // printk("[-] Execution failed!\n");
        return result;
    }

    auto output_evalue = method_ptr->get_output(0);
    Tensor output = output_evalue.toTensor();
    float* output_data = output.mutable_data_ptr<float>();

    size_t num_classes = output.numel();

    // 1. Find the maximum logit (raw score) for numerical stability
    float max_logit = output_data[0];
    int max_idx = 0;
    for(size_t i = 1; i < num_classes; i++){
        if(output_data[i] > max_logit){
            max_logit = output_data[i];
            max_idx = (int)i;
        }
    }
    result.class_id = max_idx;

    // 2. Compute Softmax to get the true probability of the predicted class
    float sum_exp = 0.0f;
    for(size_t i = 0; i < num_classes; i++){
        sum_exp += std::exp(output_data[i] - max_logit);
    }
    
    // The probability of the top class is exp(max_logit - max_logit) / sum_exp
    // Since exp(0) is 1.0, this simplifies to 1.0 / sum_exp
    float confidence_prob = 1.0f / sum_exp;
    result.confidence = confidence_prob;

    // printk("[+] Predicted class: %d with confidence: %.2f%%\n", max_idx, (double)(confidence_prob * 100.0f));

    // 3. Threshold check against true probability (0.0 to 1.0)
    if (confidence_prob >= CONFIDENCE) {
        return result;
    } else {
        // printk("[-] Prediction rejected (Confidence %.2f%% < Threshold)\n", (double)(confidence_prob * 100.0f));
        return result;
    }
}