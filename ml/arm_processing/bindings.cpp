#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

#include "includes/clip.h"
#include "includes/mfcc.h"

namespace py = pybind11;

PYBIND11_MODULE(arm_processing_bind, m) {
    m.doc() = "Python bindings for ARM CMSIS-DSP processing C++ functions";

    // -------------------------------------------------------------
    // 1. Audio Clipping Binding
    // -------------------------------------------------------------
    // Bind clip_audio operating on standard numpy float32 array in-place
        m.def("clip_audio", [](py::array_t<float, py::array::c_style | py::array::forcecast> input) {
            py::buffer_info buf = input.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Input array must be 1-dimensional");
            }
            
            // Pass standard float pointer
            return clip_audio(static_cast<float*>(buf.ptr)); 
        }, "Clips an audio buffer in-place", py::arg("input_audio"));

    // -------------------------------------------------------------
    // 2. Class Binding: MFCCExtractor (For low-level frame processing)
    // -------------------------------------------------------------
    py::class_<MFCCExtractor>(m, "MFCCExtractor")
        .def(py::init<>())
        .def("process_frame", [](MFCCExtractor& self, 
                                 py::array_t<float, py::array::c_style | py::array::forcecast> audio_in) {
            py::buffer_info in_buf = audio_in.request();
            
            if (in_buf.ndim != 1 || in_buf.shape[0] < 512) {
                throw std::runtime_error("Input audio frame must have at least FFT_SIZE (512) samples");
            }

            // Allocate output NumPy array of shape (NUM_MFCC,)
            auto result = py::array_t<float>({NUM_MFCC});
            py::buffer_info out_buf = result.request();

            self.process_frame(static_cast<const float*>(in_buf.ptr),
                               static_cast<float*>(out_buf.ptr));

            return result;
        }, "Processes a single 480-sample frame and returns a (10,) MFCC vector", py::arg("audio_in"));

    // -------------------------------------------------------------
    // 3. High-Level Helper: compute_mfcc (Matches PyTorch torchaudio.transforms.MFCC)
    // -------------------------------------------------------------
    m.def("compute_mfcc", [](py::array_t<float, py::array::c_style | py::array::forcecast> audio,
                         int hop_length = 320,
                         int n_fft = 512) {
        py::buffer_info buf = audio.request();
        
        // Support shape (N,) or (1, N)
        if (buf.ndim == 2 && buf.shape[0] == 1) {
            // Automatically accept (1, N) by treating length as shape[1]
        } else if (buf.ndim != 1) {
            throw std::runtime_error("Input audio must be 1-dimensional");
        }

        const float* audio_ptr = static_cast<const float*>(buf.ptr);
        size_t total_samples = (buf.ndim == 2) ? buf.shape[1] : buf.shape[0];

        if (total_samples < static_cast<size_t>(n_fft)) {
            throw std::runtime_error("Audio is too short for the specified FFT window size");
        }

        size_t num_frames = (total_samples - n_fft) / hop_length + 1;

        auto mfcc_output = py::array_t<float>({(size_t)NUM_MFCC, num_frames});
        py::buffer_info out_buf = mfcc_output.request();
        float* out_ptr = static_cast<float*>(out_buf.ptr);

        MFCCExtractor extractor;
        std::vector<float> temp_frame_mfcc(NUM_MFCC);

        for (size_t t = 0; t < num_frames; ++t) {
            const float* frame_start = audio_ptr + (t * hop_length);
            extractor.process_frame(frame_start, temp_frame_mfcc.data());

            for (size_t c = 0; c < NUM_MFCC; ++c) {
                out_ptr[c * num_frames + t] = temp_frame_mfcc[c];
            }
        }

        return mfcc_output;
    }, "Computes full MFCC feature matrix matching PyTorch output [NUM_MFCC, num_frames]",
    py::arg("audio"), py::arg("hop_length") = 320, py::arg("n_fft") = 512);
}