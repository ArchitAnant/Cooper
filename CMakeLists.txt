cmake_minimum_required(VERSION 3.20.0)

# --- FIX: Prevent ExecuTorch from exporting targets and crashing Zephyr ---
set(EXECUTORCH_ENABLE_INSTALL OFF CACHE INTERNAL "")
set(CMAKE_SKIP_INSTALL_RULES TRUE)
# --------------------------------------------------------------------------

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(kws_pico2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Add application and processing sources
target_sources(app PRIVATE 
    src/main.c
    src/dcnn/model_runner.cpp
    src/dcnn/model.c
    src/processing/clip.cpp
    src/processing/mfcc.cpp
    src/processing/weights.cpp
)

# 2. Add include directories (project headers & CMSIS include paths)
target_include_directories(app PRIVATE 
    src/dcnn
    src/processing/includes
    ${ZEPHYR_BASE}/../modules/lib/cmsis-dsp/Include
    ${ZEPHYR_BASE}/../modules/hal/cmsis/CMSIS/Core/Include
)

# 3. Add CMSIS-DSP compile definitions needed for fast FFT tables
target_compile_definitions(app PRIVATE
    ARM_MATH_LOOPUNROLL
    ARM_TABLE_TWIDDLECOEFS_F32
    ARM_TABLE_BITREV_1024
)

# 4. Link ExecuTorch and CMSIS-DSP
# In Zephyr, CMSIS-DSP can be linked via the Zephyr module target or direct archive
target_link_libraries(app PRIVATE 
    executorch 
    portable_ops_lib
)