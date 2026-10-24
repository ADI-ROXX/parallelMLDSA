// Correctness gate: sign with every GPU kernel and verify every signature with liboqs.
// Usage: gpu_mldsa_verify --gpu 0 --batch 2048 [--tenants 1024]
// ./benchmarks/gpu_mldsa_verify.cu
#include <oqs/oqs.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mldsa_gpu.h"
#include "reduced_gpu.h"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                                     \
            std::exit(3);                                                                \
        }                                                                                \
    } while (0)

namespace {

struct Keys {
    std::vector<uint8_t> aos;
    std::vector<uint8_t> soa;
    std::vector<uint8_t> pk;
};

// Runs one kernel over the batch and verifies every output with liboqs. Returns true on full pass.
bool check_kernel(const std::string& kernel, const Keys& keys, const std::vector<uint8_t>& msgs, OQS_SIG* sig,
                  size_t n, size_t T) {
    const int threads = 128;
    const bool naive = kernel == "naive";
    const bool reduced = kernel == "warpsign_reduced";
    const bool soa = !naive;
    const size_t scratch_words = reduced ? mldsa::reduced::SCRATCH_WORDS_R : mldsa::SCRATCH_WORDS;
    int grid = 0;
    if (reduced) grid = mldsa::reduced::warpsign_reduced_grid_blocks(threads, static_cast<int>(n));
    else if (!naive) grid = mldsa::warpsign_grid_blocks(threads, static_cast<int>(n));
    const size_t slots = naive ? n : static_cast<size_t>(grid) * threads;
    const size_t num_warps = (slots + 31) / 32;
    const std::vector<uint8_t>& sk_host = soa ? keys.soa : keys.aos;

    uint8_t *d_sk, *d_msgs, *d_sigs, *d_bytes;
    uint32_t *d_scratch, *d_attempts, *d_status, *d_queue;
    uint64_t *d_tstart, *d_tend, *d_wstats;
    CUDA_CHECK(cudaMalloc(&d_sk, sk_host.size()));
    CUDA_CHECK(cudaMalloc(&d_msgs, msgs.size()));
    CUDA_CHECK(cudaMalloc(&d_sigs, n * mldsa::SIG_LEN));
    CUDA_CHECK(cudaMalloc(&d_scratch, slots * scratch_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bytes, slots * mldsa::BYTES_PER_SLOT));
    CUDA_CHECK(cudaMalloc(&d_tstart, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_tend, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_attempts, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_status, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_wstats, 2 * num_warps * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_queue, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_sk, sk_host.data(), sk_host.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msgs, msgs.data(), msgs.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_queue, 0, sizeof(uint32_t)));

    mldsa::BatchArgs args{};
    args.sk = d_sk;
    args.soa = soa;
    args.num_tenants = static_cast<int>(T);
    args.msgs = d_msgs;
    args.sigs = d_sigs;
    args.scratch = d_scratch;
    args.bytes = d_bytes;
    args.t_start_ns = d_tstart;
    args.t_end_ns = d_tend;
    args.attempts = d_attempts;
    args.status = d_status;
    args.warp_stats = d_wstats;
    args.queue = d_queue;
    args.n = static_cast<int>(n);
    if (naive) mldsa::launch_naive(args, threads, 0);
    else if (reduced) mldsa::reduced::launch_warpsign_reduced(args, grid, threads, 0);
    else mldsa::launch_warpsign(args, grid, threads, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> sigs(n * mldsa::SIG_LEN);
    std::vector<uint32_t> status(n);
    CUDA_CHECK(cudaMemcpy(sigs.data(), d_sigs, sigs.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(status.data(), d_status, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    size_t ok = 0, bad = 0, exhausted = 0;
    for (size_t i = 0; i < n; ++i) {
        if (status[i]) { ++exhausted; ++bad; continue; }
        const size_t tenant = i % T;
        const OQS_STATUS r = OQS_SIG_verify(sig, msgs.data() + i * mldsa::MSG_LEN, mldsa::MSG_LEN,
                                            sigs.data() + i * mldsa::SIG_LEN, mldsa::SIG_LEN,
                                            keys.pk.data() + tenant * sig->length_public_key);
        if (r == OQS_SUCCESS) ++ok;
        else ++bad;
    }

    std::printf("verify: kernel=%s layout=%s batch=%zu tenants=%zu liboqs_ok=%zu failed=%zu budget_exhausted=%zu\n",
                kernel.c_str(), soa ? "soa" : "aos", n, T, ok, bad, exhausted);
    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");

    cudaFree(d_sk); cudaFree(d_msgs); cudaFree(d_sigs); cudaFree(d_scratch); cudaFree(d_bytes);
    cudaFree(d_tstart); cudaFree(d_tend); cudaFree(d_attempts); cudaFree(d_status);
    cudaFree(d_wstats); cudaFree(d_queue);
    return bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
    int gpu = 0;
    size_t n = 2048, T = 256;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--gpu") gpu = std::atoi(argv[i + 1]);
        else if (a == "--batch") n = std::strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--tenants") T = std::strtoull(argv[i + 1], nullptr, 10);
    }

    OQS_SIG* sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_44);
    if (!sig) { std::fprintf(stderr, "liboqs lacks ML-DSA-44\n"); return 2; }

    Keys keys;
    keys.aos.resize(T * sig->length_secret_key);
    keys.pk.resize(T * sig->length_public_key);
    for (size_t t = 0; t < T; ++t)
        OQS_SIG_keypair(sig, keys.pk.data() + t * sig->length_public_key,
                        keys.aos.data() + t * sig->length_secret_key);

    keys.soa.resize(keys.aos.size());
    for (size_t t = 0; t < T; ++t)
        for (size_t off = 0; off < mldsa::SK_LEN; ++off)
            keys.soa[off * T + t] = keys.aos[t * mldsa::SK_LEN + off];

    std::vector<uint8_t> msgs(n * mldsa::MSG_LEN);
    for (size_t i = 0; i < msgs.size(); ++i) msgs[i] = static_cast<uint8_t>((i * 40503u + 17) >> 3);

    CUDA_CHECK(cudaSetDevice(gpu));
    bool all_ok = true;
    for (const char* k : {"naive", "warpsign", "warpsign_reduced"})
        all_ok = check_kernel(k, keys, msgs, sig, n, T) && all_ok;

    OQS_SIG_free(sig);
    std::printf("overall: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
