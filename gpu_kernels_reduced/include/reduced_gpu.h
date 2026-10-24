// gpu_kernels_reduced/include/reduced_gpu.h
// Host interface for the reduced-scratch WarpSign kernel. A_hat is not stored: each entry is
// re-expanded from rho when it is used, which cuts per-slot scratch from 50 to 27 polynomials.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "mldsa_gpu.h"

namespace mldsa {
namespace reduced {

constexpr int SCRATCH_POLYS_R = 27;
constexpr int SCRATCH_WORDS_R = SCRATCH_POLYS_R * N;

int warpsign_reduced_grid_blocks(int threads_per_block, int n);
int launch_warpsign_reduced(const BatchArgs& args, int grid_blocks, int threads_per_block, cudaStream_t stream);

}  // namespace reduced
}  // namespace mldsa