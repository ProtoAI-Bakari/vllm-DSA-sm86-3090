# SPDX-License-Identifier: Apache-2.0
# METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
# // --ProtoAI-Bakari--
#
# CMakeLists shard for the CC3 lane sm_86 CUDA kernels.
# Include from CC2's main CMakeLists.txt:
#
#   include(${CMAKE_SOURCE_DIR}/csrc_patches/01_dsa_sm86_kernels/CMakeLists.dsa_sm86.cmake)
#
# Defines the `_dsa_sm86` Python extension target containing:
#   - sparse_attn_indexer_sm86.cu       (CC3 main indexer logits kernel)
#   - sparse_attn_indexer_sm86_tuned.cu (CC3 tuned variant w/ smem opt-in)
#   - fp8_kv_dequant_sm86.cu            (CC3 DSA-v4 fp8 -> bf16)
#   - torch_bindings_dsa_sm86.cpp       (torch.ops registration)
#
# Requirements:
#   - CUDA >= 11.0 (mma.sync.aligned.m16n8k16)
#   - CUDA_ARCHS must include 8.6 (or 8.0/8.7 — sm_80+ async-copy & MMA path)
#   - Torch headers + pybind11 from the parent build

set(DSA_SM86_KERNELS_DIR
    "${CMAKE_CURRENT_LIST_DIR}")

set(DSA_SM86_SOURCES
    "${DSA_SM86_KERNELS_DIR}/sparse_attn_indexer_sm86.cu"
    "${DSA_SM86_KERNELS_DIR}/sparse_attn_indexer_sm86_tuned.cu"
    "${DSA_SM86_KERNELS_DIR}/fp8_kv_dequant_sm86.cu"
    "${DSA_SM86_KERNELS_DIR}/torch_bindings_dsa_sm86.cpp"
)

# Verify all source files exist (catch path drift early in build).
foreach(src ${DSA_SM86_SOURCES})
    if(NOT EXISTS "${src}")
        message(FATAL_ERROR
                "DSA sm_86 source missing: ${src} — has CC3 lane been pulled?")
    endif()
endforeach()

# Filter the parent CUDA_ARCHS down to the sm_80+ subset that supports our
# inline PTX (cp.async.cg, mma.sync.aligned.m16n8k16, ldmatrix.sync.aligned).
set(DSA_SM86_SUPPORT_ARCHS "8.0;8.6;8.7;8.9;9.0a;10.0a;10.0f")
if(DEFINED CUDA_ARCHS)
    set(DSA_SM86_BUILD_ARCHS "")
    foreach(arch ${CUDA_ARCHS})
        list(FIND DSA_SM86_SUPPORT_ARCHS "${arch}" idx)
        if(idx GREATER -1)
            list(APPEND DSA_SM86_BUILD_ARCHS "${arch}")
        endif()
    endforeach()
else()
    set(DSA_SM86_BUILD_ARCHS "8.6")  # Project target: RTX 3090
endif()

if(NOT DSA_SM86_BUILD_ARCHS)
    message(STATUS
            "DSA sm_86 kernels skipped: no compatible CUDA_ARCHS "
            "(need >= 8.0). _dsa_sm86 target will be empty stub.")
    add_custom_target(_dsa_sm86)
    return()
endif()

message(STATUS "DSA sm_86 kernels: building for archs ${DSA_SM86_BUILD_ARCHS}")

# Compile flags — match the surrounding vLLM build settings + sm_86 minimum.
set(DSA_SM86_GPU_FLAGS
    "${VLLM_GPU_FLAGS}"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "-std=c++17"
    "-Xcompiler=-fPIC"
)

# Per-source gencode flags (CC2's helper — same one used by FlashMLA shard).
if(COMMAND set_gencode_flags_for_srcs)
    set_gencode_flags_for_srcs(
        SRCS "${DSA_SM86_SOURCES}"
        CUDA_ARCHS "${DSA_SM86_BUILD_ARCHS}")
endif()

# Define the extension target through CC2's helper if available; otherwise
# emit a placeholder add_library() that CC2 can wire into the actual install.
if(COMMAND define_extension_target)
    define_extension_target(
        _dsa_sm86
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${DSA_SM86_SOURCES}
        COMPILE_FLAGS ${DSA_SM86_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/csrc"
                            "${CMAKE_SOURCE_DIR}/csrc_patches/01_dsa_sm86_kernels"
        USE_SABI 3
        WITH_SOABI)

    # Disable Stable ABI for CUDA + C++ files (matches FlashMLA recipe so
    # nvcc can use full Py types without Py_LIMITED_API restrictions).
    target_compile_options(_dsa_sm86 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
else()
    # Fallback for ad-hoc builds that don't use CC2's helper macros.
    add_library(_dsa_sm86 MODULE ${DSA_SM86_SOURCES})
    target_include_directories(_dsa_sm86 PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc"
        "${DSA_SM86_KERNELS_DIR}")
    set_target_properties(_dsa_sm86 PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_dsa_sm86"
        CUDA_SEPARABLE_COMPILATION OFF)
endif()

# // --ProtoAI-Bakari--
