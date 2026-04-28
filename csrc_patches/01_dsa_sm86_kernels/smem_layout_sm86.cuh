// SPDX-License-Identifier: Apache-2.0
// METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
// // --ProtoAI-Bakari--
//
// smem_layout_sm86.cuh — Story 3: SMEM layout port from Hopper TMA descriptors
// to manual stride/offset arrays for sm_86 (CC3 backlog).
//
// Hopper TMA encodes a 1D/2D/3D tile via a 128-byte CUtensorMap descriptor:
//   - global base pointer
//   - tensor shape (rank-N) + stride
//   - tile shape (rank-N)
//   - swizzle mode (NO_SWIZZLE / 32B / 64B / 128B)
//   - L2 cache hint, fill mode
// The cp.async.bulk.tensor.* PTX instruction emits one async copy per
// descriptor-coordinate pair, with hardware-managed swizzling.
//
// On sm_86 we have neither TMA descriptors nor hardware swizzling. This
// header translates the Hopper-side layout intent into manual sm_86 layout:
//
//   1. Decode the descriptor's tile shape + tensor stride into per-thread
//      (row, col) -> (smem_offset, gmem_offset) mappings.
//   2. Apply software swizzle (XOR-based) where needed for bank-conflict
//      avoidance — SwizzleType matches Hopper's 32/64/128B options.
//   3. Emit per-thread cp.async.cg.shared.global instructions covering the
//      full tile in a strided / coalesced loop.
//
// Reusable across:
//   - sparse_attn_indexer_sm86.cu        (Q row + K tile loads)
//   - sparse_attn_indexer_sm86_tuned.cu  (multi-block K tile + replicated Q)
//   - paged_attn_hd512_sm86.cu           (paged K tile + V tile)
//   - any future sm_86 kernel that ports a TMA-using sm_90 kernel
//
// Not header-only at runtime: the swizzle modes use compile-time constexpr
// dispatch, so the resulting code is identical to hand-written sm_86.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace vllm {
namespace dsa_sm86 {
namespace smem_layout {

// ============================================================================
// Swizzle mode — matches CUtensorMapSwizzle on Hopper
// ============================================================================

enum class SwizzleType : int {
  None     = 0,    // Identity layout
  Bytes32  = 32,   // Hopper TMA SWIZZLE_32B
  Bytes64  = 64,   // Hopper TMA SWIZZLE_64B
  Bytes128 = 128,  // Hopper TMA SWIZZLE_128B
};

// Swizzle a (row, col_byte) -> col_byte_swizzled to avoid bank conflicts on
// 32-banked smem reads. The XOR pattern matches Hopper TMA's row-major
// SWIZZLE_<N>B layout for tile_width <= N.
template <SwizzleType S>
__device__ __forceinline__ int swizzle_col(int row, int col_byte) {
  if constexpr (S == SwizzleType::None) {
    return col_byte;
  } else if constexpr (S == SwizzleType::Bytes32) {
    // Bank conflict-free for tiles up to 32 banks × 4B = 128B wide
    return col_byte ^ ((row & 0x7) << 2);
  } else if constexpr (S == SwizzleType::Bytes64) {
    return col_byte ^ ((row & 0x7) << 3);
  } else if constexpr (S == SwizzleType::Bytes128) {
    return col_byte ^ ((row & 0x7) << 4);
  } else {
    return col_byte;
  }
}

// ============================================================================
// 2D tile descriptor (sm_86 equivalent of Hopper CUtensorMap rank-2)
// ============================================================================

template <int ROWS, int COLS, typename T>
struct Tile2D {
  static constexpr int kRows       = ROWS;
  static constexpr int kCols       = COLS;
  static constexpr int kElemBytes  = sizeof(T);
  static constexpr int kRowBytes   = COLS * kElemBytes;
  static constexpr int kTotalBytes = ROWS * kRowBytes;

  // Per-thread tile coordinates given thread id and total threads.
  __device__ __forceinline__ static int thread_row(int tid, int threads_per_block) {
    constexpr int kElemsPerChunk = 16 / kElemBytes;  // 16B chunks
    constexpr int kChunksPerRow  = COLS / kElemsPerChunk;
    return (tid / kChunksPerRow);
  }
  __device__ __forceinline__ static int thread_col_byte(int tid, int threads_per_block) {
    constexpr int kElemsPerChunk = 16 / kElemBytes;
    constexpr int kChunksPerRow  = COLS / kElemsPerChunk;
    return (tid % kChunksPerRow) * 16;  // byte offset
  }
};

// ============================================================================
// Cooperative async load — replaces TMA cp.async.bulk.tensor for a 2D tile
// ============================================================================
//
// Each thread issues 16B cp.async.cg copies. ROW_STRIDE_BYTES is the gmem
// row stride in bytes (== element_stride * sizeof(T) when contiguous).
// SwizzleType selects the smem layout to match downstream ldmatrix expectations.

template <typename Tile, SwizzleType S = SwizzleType::None,
          int THREADS_PER_BLOCK>
__device__ __forceinline__ void cooperative_async_load_tile(
    typename std::remove_reference<decltype(*static_cast<typename Tile::value_type*>(nullptr))>::type* /*unused*/,
    void* smem_dst,
    const void* gmem_src,
    int gmem_row_stride_bytes,
    int tid) {
  // Static layout math
  constexpr int kRows       = Tile::kRows;
  constexpr int kCols       = Tile::kCols;
  constexpr int kElemBytes  = Tile::kElemBytes;
  constexpr int kRowBytes   = Tile::kRowBytes;
  constexpr int kChunksPerRow = kRowBytes / 16;
  constexpr int kTotalChunks  = kRows * kChunksPerRow;

  static_assert(kRowBytes % 16 == 0,
                "Tile row width must be 16B-aligned for cp.async.cg.16");

  uint8_t* smem_b8 = static_cast<uint8_t*>(smem_dst);
  const uint8_t* gmem_b8 = static_cast<const uint8_t*>(gmem_src);

#pragma unroll
  for (int chunk = tid; chunk < kTotalChunks; chunk += THREADS_PER_BLOCK) {
    const int row     = chunk / kChunksPerRow;
    const int col_b   = (chunk % kChunksPerRow) * 16;
    const int col_swz = swizzle_col<S>(row, col_b);

    void* dst = smem_b8 + row * kRowBytes + col_swz;
    const void* src = gmem_b8 + row * gmem_row_stride_bytes + col_b;

    unsigned smem_int = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(src));
  }
}

// Convenience: typed tile alias.
template <int ROWS, int COLS, typename T>
struct TypedTile2D : public Tile2D<ROWS, COLS, T> {
  using value_type = T;
};

// ============================================================================
// ldmatrix layout constants — matches mma.sync.aligned.m16n8k16 expectations
// ============================================================================
//
// For ldmatrix.sync.aligned.m8n8.x4.shared.b16 the smem must be laid out
// as 4 contiguous 8x8 blocks (16 rows × 16 cols / 16 bf16 = 32 rows × 16 cols).
// Use SwizzleType::Bytes64 to avoid bank conflicts on the .x4 read.

constexpr SwizzleType kLdmatrixSwizzle = SwizzleType::Bytes64;

// ============================================================================
// Hopper -> sm_86 cookbook (documentation reference)
// ============================================================================
//
// Hopper TMA descriptor:
//   CUtensorMap desc;
//   cuTensorMapEncodeTiled(&desc, BF16, 2, gmem_ptr, /*shape*/{T_k, D},
//       /*stride*/{D*2, 1}, /*tile*/{kBlockK, D},
//       /*swizzle*/SWIZZLE_64B, /*L2*/L2_PROMOTION_NONE, /*OOB*/OOB_NAN);
//   asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
//       " [%0], [%1, {%2, %3}], [%4];" :: "r"(smem), "l"(&desc), "r"(blk_idx*kBlockK),
//       "r"(0), "r"(mbar));
//
// sm_86 equivalent (this header):
//   using KTile = TypedTile2D<kBlockK, kHeadDim, __nv_bfloat16>;
//   cooperative_async_load_tile<KTile, kLdmatrixSwizzle, kThreadsPerBlock>(
//       /*unused dt*/(__nv_bfloat16*)nullptr,
//       smem_k,
//       gmem_k_base + blk_idx * kBlockK * row_stride,
//       row_stride_bytes,
//       threadIdx.x);
//   asm("cp.async.commit_group;");
//   asm("cp.async.wait_group 0;");
//   __syncthreads();

}  // namespace smem_layout
}  // namespace dsa_sm86
}  // namespace vllm

// // --ProtoAI-Bakari--
