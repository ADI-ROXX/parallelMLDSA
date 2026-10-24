// GPU benchmark: ML-DSA-44 batch signing with the naive or WarpSign kernel.
// Usage: gpu_mldsa_bench --kernel naive|warpsign --gpu 0 --batch 10000 --tier mldsa44
//                        [--trials 5] [--tenants 1024] [--threads 128] [--layout aos|soa] [--csv]
// ./benchmarks/gpu_mldsa_bench.cu
#include <oqs/oqs.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

struct Options {
    std::string kernel = "warpsign";
    int gpu = 0;
    size_t batch = 10000;
    std::string tier = "mldsa44";
    int trials = 5;
    size_t tenants = 1024;
    int threads = 128;
    std::string layout;  // empty = kernel default
    bool csv = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--kernel") o.kernel = next();
        else if (a == "--gpu") o.gpu = std::atoi(next());
        else if (a == "--batch") o.batch = std::strtoull(next(), nullptr, 10);
        else if (a == "--tier") o.tier = next();
        else if (a == "--trials") o.trials = std::atoi(next());
        else if (a == "--tenants") o.tenants = std::strtoull(next(), nullptr, 10);
        else if (a == "--threads") o.threads = std::atoi(next());
        else if (a == "--layout") o.layout = next();
        else if (a == "--csv") o.csv = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); std::exit(2); }
    }
    return o;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    if (opt.tier != "mldsa44") {
        std::fprintf(stderr, "GPU kernels currently implement mldsa44 only\n");
        return 2;
    }
    const bool naive = opt.kernel == "naive";
    const bool reduced = opt.kernel == "warpsign_reduced";
    const bool persistent = opt.kernel == "warpsign" || reduced;
    if (!naive && !persistent) {
        std::fprintf(stderr, "kernel must be naive, warpsign or warpsign_reduced\n");
        return 2;
    }
    const bool use_soa = opt.layout.empty() ? persistent : (opt.layout == "soa");
    const size_t scratch_words = reduced ? mldsa::reduced::SCRATCH_WORDS_R : mldsa::SCRATCH_WORDS;

    CUDA_CHECK(cudaSetDevice(opt.gpu));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.gpu));

    // Keygen: one ML-DSA-44 key per tenant, AoS on the host.
    OQS_SIG* sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_44);
    if (!sig) { std::fprintf(stderr, "liboqs lacks ML-DSA-44\n"); return 2; }
    const size_t T = opt.tenants;
    std::vector<uint8_t> sk_aos(T * sig->length_secret_key);
    std::vector<uint8_t> pk(T * sig->length_public_key);
    for (size_t t = 0; t < T; ++t)
        OQS_SIG_keypair(sig, pk.data() + t * sig->length_public_key, sk_aos.data() + t * sig->length_secret_key);
    OQS_SIG_free(sig);

    // Key layout: SoA transposes so that byte `off` of every tenant is contiguous.
    std::vector<uint8_t> sk_host(sk_aos.size());
    if (use_soa) {
        for (size_t t = 0; t < T; ++t)
            for (size_t off = 0; off < mldsa::SK_LEN; ++off)
                sk_host[off * T + t] = sk_aos[t * mldsa::SK_LEN + off];
    } else {
        sk_host = sk_aos;
    }

    const size_t n = opt.batch;
    std::vector<uint8_t> msgs(n * mldsa::MSG_LEN);
    for (size_t i = 0; i < msgs.size(); ++i) msgs[i] = static_cast<uint8_t>((i * 2654435761u) >> 24);

    // Device buffers
    const int threads = opt.threads;
    int grid_blocks = 0;
    size_t slots = 0;
    if (persistent) {
        grid_blocks = reduced ? mldsa::reduced::warpsign_reduced_grid_blocks(threads, static_cast<int>(n))
                              : mldsa::warpsign_grid_blocks(threads, static_cast<int>(n));
        slots = static_cast<size_t>(grid_blocks) * threads;
    } else {
        slots = mldsa::naive_slots(static_cast<int>(n));
    }
    const size_t num_warps = (slots + 31) / 32;

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

    mldsa::BatchArgs args{};
    args.sk = d_sk;
    args.soa = use_soa;
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

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(d_queue, 0, sizeof(uint32_t)));
        if (naive) mldsa::launch_naive(args, threads, 0);
        else if (reduced) mldsa::reduced::launch_warpsign_reduced(args, grid_blocks, threads, 0);
        else mldsa::launch_warpsign(args, grid_blocks, threads, 0);
    };

    // Warmup run (not recorded) absorbs JIT and first-touch costs.
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_attempts(n), h_status(n);
    std::vector<uint64_t> h_tstart(n), h_tend(n);
    std::vector<uint64_t> h_wstats(2 * num_warps);

    for (int trial = 0; trial < opt.trials; ++trial) {
        CUDA_CHECK(cudaEventRecord(ev0, 0));
        launch();
        CUDA_CHECK(cudaEventRecord(ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

        CUDA_CHECK(cudaMemcpy(h_attempts.data(), d_attempts, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_status.data(), d_status, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tstart.data(), d_tstart, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tend.data(), d_tend, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_wstats.data(), d_wstats, 2 * num_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        std::vector<double> lat_us(n);
        for (size_t i = 0; i < n; ++i) lat_us[i] = static_cast<double>(h_tend[i] - h_tstart[i]) / 1000.0;
        const double mean_us = std::accumulate(lat_us.begin(), lat_us.end(), 0.0) / n;
        const double p50 = percentile(lat_us, 0.50);
        const double p99 = percentile(lat_us, 0.99);
        const double throughput = static_cast<double>(n) / (ms / 1000.0);

        const double mean_attempts =
            std::accumulate(h_attempts.begin(), h_attempts.end(), 0.0, [](double s, uint32_t v) { return s + v; }) / n;

        // Warp execution efficiency: useful lane-iterations / (32 * issued warp iterations).
        double warp_eff = 0.0;
        if (naive) {
            double useful = 0.0, issued = 0.0;
            for (size_t w = 0; w * 32 < n; ++w) {
                uint32_t mx = 0;
                size_t lanes = std::min<size_t>(32, n - w * 32);
                for (size_t l = 0; l < lanes; ++l) {
                    useful += h_attempts[w * 32 + l];
                    mx = std::max(mx, h_attempts[w * 32 + l]);
                }
                issued += 32.0 * mx;
            }
            warp_eff = issued > 0 ? useful / issued : 0.0;
        } else {
            double useful = 0.0, issued = 0.0;
            for (size_t w = 0; w < num_warps; ++w) {
                useful += static_cast<double>(h_wstats[2 * w + 1]);
                issued += 32.0 * static_cast<double>(h_wstats[2 * w]);
            }
            warp_eff = issued > 0 ? useful / issued : 0.0;
        }

        size_t failures = 0;
        for (uint32_t s : h_status) failures += s;

        const char* kname = naive ? "naive" : (reduced ? "warpsign_reduced" : "warpsign");
        if (opt.csv) {
            std::printf("gpu,%s,%s,%s,%zu,%zu,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%zu,%s\n", kname, prop.name,
                        opt.tier.c_str(), n, T, trial, ms, throughput, mean_us, p50, p99, mean_attempts,
                        warp_eff, failures, use_soa ? "soa" : "aos");
        } else {
            std::printf("[gpu %s %s batch=%zu trial=%d] %.1f sig/s kernel=%.3fms mean=%.1fus p50=%.1fus p99=%.1fus "
                        "attempts=%.3f warp_eff=%.3f failures=%zu layout=%s\n",
                        kname, prop.name, n, trial, throughput, ms, mean_us, p50, p99, mean_attempts, warp_eff,
                        failures, use_soa ? "soa" : "aos");
        }
        std::fflush(stdout);
    }

    cudaFree(d_sk);
    cudaFree(d_msgs);
    cudaFree(d_sigs);
    cudaFree(d_scratch);
    cudaFree(d_bytes);
    cudaFree(d_tstart);
    cudaFree(d_tend);
    cudaFree(d_attempts);
    cudaFree(d_status);
    cudaFree(d_wstats);
    cudaFree(d_queue);
    return 0;
}
