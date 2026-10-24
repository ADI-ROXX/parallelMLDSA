// Host-side interface to the GPU ML-DSA-44 batch signing kernels.
// gpu_kernels/include/mldsa_gpu.h

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "common_mldsa.cuh"

namespace mldsa {

struct BatchArgs {
    const uint8_t* sk;        // key store: AoS (tenant-major) or SoA (offset-major)
    bool soa;
    int num_tenants;
    const uint8_t* msgs;      // n * MSG_LEN
    uint8_t* sigs;            // n * SIG_LEN output
    uint32_t* scratch;        // num_slots * SCRATCH_WORDS
    uint8_t* bytes;           // num_slots * BYTES_PER_SLOT
    uint64_t* t_start_ns;     // n, per-request start (globaltimer)
    uint64_t* t_end_ns;       // n, per-request completion
    uint32_t* attempts;       // n, rejection-sampling attempts per request
    uint32_t* status;         // n, 0 = signed, 1 = attempt budget exhausted
    uint64_t* warp_stats;     // 2 * num_warps: [iterations, active-lane sum], warpsign only
    uint32_t* queue;          // global work counter, warpsign only; must be zeroed before launch
    int n;
};

// Slots needed for the naive kernel (one per request) and the persistent warpsign kernel.
inline size_t naive_slots(int n) { return static_cast<size_t>(n); }

// Launch one thread per request. Warp efficiency is derived from the attempts array.
void launch_naive(const BatchArgs& args, int threads_per_block, cudaStream_t stream);

// Persistent-warp kernel with in-warp refill of finished lanes. Returns the number of slots
// it uses (grid_blocks * threads_per_block); scratch must be allocated for that many slots.
int launch_warpsign(const BatchArgs& args, int grid_blocks, int threads_per_block, cudaStream_t stream);

// Persistent grid sized so each thread serves several requests; capped at device occupancy.
int warpsign_grid_blocks(int threads_per_block, int n);

}  // namespace mldsa