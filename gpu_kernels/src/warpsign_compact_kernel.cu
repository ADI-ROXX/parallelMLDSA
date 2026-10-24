// WarpSign: persistent warps with per-iteration compaction. After every rejection attempt the
// warp votes (__ballot_sync) on which lanes produced a signature; those lanes write their output
// and immediately pop the next request, so lanes never idle while other lanes in the warp retry.
// Keys are read from the SoA layout so that lanes at the same key offset coalesce.
// gpu_kernels/src/warpsign_compact_kernel.cu
#include "mldsa_gpu.h"
#include "mldsa_sign_device.cuh"

namespace mldsa {

__device__ __forceinline__ uint64_t now_ns_ws() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void warpsign_kernel(BatchArgs a) {
    const unsigned FULL = 0xFFFFFFFFu;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = gtid >> 5;

    PolySlot S = warp_slot<uint32_t, SCRATCH_WORDS>(a.scratch, gtid);
    ByteSlot B = warp_slot<uint8_t, BYTES_PER_SLOT>(a.bytes, gtid);
    const KeyView kv{a.sk, a.soa, a.num_tenants};

    bool active = false;
    bool pending_prep = false;
    int req = 0;
    uint32_t kappa = 0;
    uint32_t tries = 0;
    uint64_t t0 = 0;

    req = static_cast<int>(atomicAdd(a.queue, 1u));
    if (req < a.n) {
        t0 = now_ns_ws();
        mldsa_prepare(S, B, kv, req % a.num_tenants, a.msgs + static_cast<size_t>(req) * MSG_LEN);
        active = true;
    }

    uint64_t iters = 0;
    uint64_t active_sum = 0;

    // Amortized warp-compaction loop: finished lanes pop their next request immediately and mark
    // `pending_prep`. Expensive key/message preparation runs only when >= 8 lanes in the warp are
    // waiting to prepare (or no lanes remain in the attempt stage), slashing refill branch stalls.
    while (__ballot_sync(FULL, active || pending_prep)) {
        const unsigned prep_mask = __ballot_sync(FULL, pending_prep);
        const unsigned act_mask = __ballot_sync(FULL, active);
        if (prep_mask && (__popc(prep_mask) >= 8 || act_mask == 0u)) {
            if (pending_prep) {
                t0 = now_ns_ws();
                mldsa_prepare(S, B, kv, req % a.num_tenants, a.msgs + static_cast<size_t>(req) * MSG_LEN);
                pending_prep = false;
                active = true;
            }
            __syncwarp(FULL);
        }

        const int nactive = __popc(__ballot_sync(FULL, active));
        if (nactive == 0) continue;

        bool accepted = false;
        if (active) {
            ++tries;
            uint8_t* sig = a.sigs + static_cast<size_t>(req) * SIG_LEN;
            accepted = mldsa_attempt(S, B, sig, kappa);
            if (!accepted) {
                kappa += L;
                if (tries >= MAX_ATTEMPTS) accepted = true;
            }
        }

        if (lane == 0) {
            ++iters;
            active_sum += static_cast<uint64_t>(nactive);
        }

        if (active && accepted) {
            const uint64_t t1 = now_ns_ws();
            a.t_start_ns[req] = t0;
            a.t_end_ns[req] = t1;
            a.attempts[req] = tries;
            a.status[req] = tries >= MAX_ATTEMPTS ? 1u : 0u;

            req = static_cast<int>(atomicAdd(a.queue, 1u));
            kappa = 0;
            tries = 0;
            active = false;
            if (req < a.n) {
                pending_prep = true;
            }
        }
        __syncwarp(FULL);
    }

    if (lane == 0) {
        a.warp_stats[2 * warp_id] = iters;
        a.warp_stats[2 * warp_id + 1] = active_sum;
    }
}

// Fill the device with resident persistent blocks; batches larger than the resulting slot count refill via the queue.
int warpsign_grid_blocks(int threads_per_block, int n) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, warpsign_kernel, threads_per_block, 0);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    const int fill = prop.multiProcessorCount * blocks_per_sm;
    const int needed = (n + threads_per_block - 1) / threads_per_block;
    if (needed < 1) return 1;
    return needed < fill ? needed : fill;
}

// c_zetas is per translation unit, so each kernel file uploads its own copy once.
int launch_warpsign(const BatchArgs& args, int grid_blocks, int threads_per_block, cudaStream_t stream) {
    static bool tables_uploaded = false;
    if (!tables_uploaded) {
        upload_tables();
        tables_uploaded = true;
    }
    warpsign_kernel<<<grid_blocks, threads_per_block, 0, stream>>>(args);
    return grid_blocks * threads_per_block;
}

}  // namespace mldsa