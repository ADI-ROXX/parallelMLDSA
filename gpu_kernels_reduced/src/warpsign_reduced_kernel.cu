// gpu_kernels_reduced/src/warpsign_reduced_kernel.cu
// WarpSign with reduced per-slot scratch (27 polynomials instead of 50). Same persistent-warp,
// __ballot_sync compaction and atomic queue as gpu_kernels/src/warpsign_compact_kernel.cu.
#include "reduced_device.cuh"

namespace mldsa {
namespace reduced {

__device__ __forceinline__ uint64_t now_ns_r() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void warpsign_reduced_kernel(BatchArgs a) {
    const unsigned FULL = 0xFFFFFFFFu;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = gtid >> 5;

    uint32_t* S = a.scratch + static_cast<size_t>(gtid) * SCRATCH_WORDS_R;
    uint8_t* B = a.bytes + static_cast<size_t>(gtid) * BYTES_PER_SLOT;
    const KeyView kv{a.sk, a.soa, a.num_tenants};

    bool active = false;
    int req = 0;
    uint32_t kappa = 0;
    uint32_t tries = 0;
    uint64_t t0 = 0;

    req = static_cast<int>(atomicAdd(a.queue, 1u));
    if (req < a.n) {
        t0 = now_ns_r();
        prepare_r(S, B, kv, req % a.num_tenants, a.msgs + static_cast<size_t>(req) * MSG_LEN);
        active = true;
    }

    uint64_t iters = 0;
    uint64_t active_sum = 0;

    while (__ballot_sync(FULL, active)) {
        bool accepted = false;
        if (active) {
            ++tries;
            uint8_t* sig = a.sigs + static_cast<size_t>(req) * SIG_LEN;
            accepted = attempt_r(S, B, sig, kappa);
            if (!accepted) {
                kappa += L;
                if (tries >= MAX_ATTEMPTS) accepted = true;
            }
        }

        const int nactive = __popc(__ballot_sync(FULL, active));
        if (lane == 0) {
            ++iters;
            active_sum += static_cast<uint64_t>(nactive);
        }

        if (active && accepted) {
            const uint64_t t1 = now_ns_r();
            a.t_start_ns[req] = t0;
            a.t_end_ns[req] = t1;
            a.attempts[req] = tries;
            a.status[req] = tries >= MAX_ATTEMPTS ? 1u : 0u;

            req = static_cast<int>(atomicAdd(a.queue, 1u));
            kappa = 0;
            tries = 0;
            if (req < a.n) {
                t0 = now_ns_r();
                prepare_r(S, B, kv, req % a.num_tenants, a.msgs + static_cast<size_t>(req) * MSG_LEN);
            } else {
                active = false;
            }
        }
        __syncwarp(FULL);
    }

    if (lane == 0) {
        a.warp_stats[2 * warp_id] = iters;
        a.warp_stats[2 * warp_id + 1] = active_sum;
    }
}

int warpsign_reduced_grid_blocks(int threads_per_block, int n) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, warpsign_reduced_kernel, threads_per_block, 0);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    const int fill = prop.multiProcessorCount * blocks_per_sm;
    const int needed = (n + threads_per_block - 1) / threads_per_block;
    if (needed < 1) return 1;
    return needed < fill ? needed : fill;
}

int launch_warpsign_reduced(const BatchArgs& args, int grid_blocks, int threads_per_block, cudaStream_t stream) {
    static bool tables_uploaded = false;
    if (!tables_uploaded) {
        upload_tables();
        tables_uploaded = true;
    }
    warpsign_reduced_kernel<<<grid_blocks, threads_per_block, 0, stream>>>(args);
    return grid_blocks * threads_per_block;
}

}  // namespace reduced
}  // namespace mldsa