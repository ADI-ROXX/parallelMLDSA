// Baseline: one thread per signature. Each thread runs its own rejection loop, so lanes in a
// warp idle while the slowest lane in that warp finishes.
// gpu_kernels/src/naive_batch_kernel.cu
#include "mldsa_gpu.h"
#include "mldsa_sign_device.cuh"

namespace mldsa {

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void naive_sign_kernel(BatchArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n) return;

    PolySlot S = warp_slot<uint32_t, SCRATCH_WORDS>(a.scratch, i);
    ByteSlot B = warp_slot<uint8_t, BYTES_PER_SLOT>(a.bytes, i);
    const KeyView kv{a.sk, a.soa, a.num_tenants};

    const uint64_t t0 = now_ns();
    mldsa_prepare(S, B, kv, i % a.num_tenants, a.msgs + static_cast<size_t>(i) * MSG_LEN);

    uint32_t kappa = 0;
    uint32_t tries = 0;
    bool ok = false;
    uint8_t* sig = a.sigs + static_cast<size_t>(i) * SIG_LEN;
    while (tries < MAX_ATTEMPTS) {
        ++tries;
        if (mldsa_attempt(S, B, sig, kappa)) {
            ok = true;
            break;
        }
        kappa += L;
    }
    const uint64_t t1 = now_ns();

    a.t_start_ns[i] = t0;
    a.t_end_ns[i] = t1;
    a.attempts[i] = tries;
    a.status[i] = ok ? 0u : 1u;
}

// c_zetas is per translation unit, so each kernel file uploads its own copy once.
void launch_naive(const BatchArgs& args, int threads_per_block, cudaStream_t stream) {
    static bool tables_uploaded = false;
    if (!tables_uploaded) {
        upload_tables();
        tables_uploaded = true;
    }
    const int blocks = (args.n + threads_per_block - 1) / threads_per_block;
    naive_sign_kernel<<<blocks, threads_per_block, 0, stream>>>(args);
}

}  // namespace mldsa