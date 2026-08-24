import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

zephyr_base = "/Volumes/Ari_Dev/zephyr_proj"

# CMSIS Include Paths
cmsis_dsp_include = os.path.join(zephyr_base, "modules/lib/cmsis-dsp/Include")
cmsis_core_include = os.path.join(zephyr_base, "modules/hal/cmsis/CMSIS/Core/Include")
cmsis_dsp_src = os.path.join(zephyr_base, "modules/lib/cmsis-dsp/Source")

# Collect all float32/common files safely
required_cmsis_files = [
    # Transforms & Inits (FFT tables and structures)
    os.path.join(cmsis_dsp_src, "TransformFunctions", "arm_rfft_fast_f32.c"),
    os.path.join(cmsis_dsp_src, "TransformFunctions", "arm_rfft_fast_init_f32.c"),
    os.path.join(cmsis_dsp_src, "TransformFunctions", "arm_cfft_f32.c"),
    os.path.join(cmsis_dsp_src, "TransformFunctions", "arm_cfft_init_f32.c"),     # <-- Fixes _arm_cfft_init_1024_f32
    os.path.join(cmsis_dsp_src, "TransformFunctions", "arm_cfft_radix8_f32.c"),
    
    # Common Tables
    os.path.join(cmsis_dsp_src, "CommonTables", "arm_common_tables.c"),
    os.path.join(cmsis_dsp_src, "CommonTables", "arm_const_structs.c"),
    
    # Complex Math
    os.path.join(cmsis_dsp_src, "ComplexMathFunctions", "arm_cmplx_mag_squared_f32.c"),
    
    # Matrix Functions
    os.path.join(cmsis_dsp_src, "MatrixFunctions", "arm_mat_mult_f32.c"),
    os.path.join(cmsis_dsp_src, "MatrixFunctions", "arm_mat_init_f32.c"),
    
    # Basic Math
    os.path.join(cmsis_dsp_src, "BasicMathFunctions", "arm_mult_f32.c"),
    os.path.join(cmsis_dsp_src, "BasicMathFunctions", "arm_scale_f32.c"),
    
    # Fast Math
    os.path.join(cmsis_dsp_src, "FastMathFunctions", "arm_vlog_f32.c"),
]

# Add bitreversal file (either arm_bitreversal2.c or arm_bitreversal.c)
for bitrev in ["arm_bitreversal2.c", "arm_bitreversal.c"]:
    p = os.path.join(cmsis_dsp_src, "TransformFunctions", bitrev)
    if os.path.exists(p):
        required_cmsis_files.append(p)
        break

# Filter out any non-existent paths just in case
required_cmsis_files = [f for f in required_cmsis_files if os.path.exists(f)]

local_sources = ['bindings.cpp', 'clip.cpp', 'mfcc.cpp', 'weights.cpp']
all_sources = local_sources + required_cmsis_files

setup(
    name='arm_processing_bind',
    ext_modules=[
        CppExtension(
            name='arm_processing_bind',
            sources=all_sources,
            include_dirs=[
                os.path.abspath('includes'),
                cmsis_dsp_include,
                cmsis_core_include,
            ],
            extra_compile_args=[
                '-std=c++17', 
                '-O3', 
                '-DARM_MATH_LOOPUNROLL',
                '-DARM_TABLE_TWIDDLECOEFS_F32',
                '-DARM_TABLE_BITREV_1024',
                '-Wno-c++11-narrowing',
                '-Wno-deprecated'
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)