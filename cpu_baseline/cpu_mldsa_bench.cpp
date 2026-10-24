// CPU baseline: multi-threaded ML-DSA signing via liboqs.
// Usage: cpu_mldsa_bench --threads 32 --batch 10000 --tier mldsa44 --trials 5 [--tenants 1024] [--csv]
// cpu_baseline/cpu_mldsa_bench.cpp

#include <oqs/oqs.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>

#include "db_mock_fetcher.hpp"

namespace {

struct Options {
    int threads = 32;
    size_t batch = 10000;
    std::string tier = "mldsa44";
    int trials = 5;
    size_t tenants = 1024;
    size_t msg_len = 64;
    bool csv = false;
};

const char* oqs_alg_for(const std::string& tier) {
    if (tier == "mldsa44") return OQS_SIG_alg_ml_dsa_44;
    if (tier == "mldsa65") return OQS_SIG_alg_ml_dsa_65;
    if (tier == "mldsa87") return OQS_SIG_alg_ml_dsa_87;
    return nullptr;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--threads") o.threads = std::atoi(next());
        else if (a == "--batch") o.batch = std::strtoull(next(), nullptr, 10);
        else if (a == "--tier") o.tier = next();
        else if (a == "--trials") o.trials = std::atoi(next());
        else if (a == "--tenants") o.tenants = std::strtoull(next(), nullptr, 10);
        else if (a == "--csv") o.csv = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); std::exit(2); }
    }
    return o;
}

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

TenantKeyStore build_keystore(OQS_SIG* sig, size_t tenants) {
    TenantKeyStore ks;
    ks.num_tenants = tenants;
    ks.sk_len = sig->length_secret_key;
    ks.pk_len = sig->length_public_key;
    ks.sks.resize(tenants * ks.sk_len);
    ks.pks.resize(tenants * ks.pk_len);
    #pragma omp parallel for schedule(dynamic)
    for (long t = 0; t < static_cast<long>(tenants); ++t) {
        OQS_SIG_keypair(sig, ks.pks.data() + t * ks.pk_len, ks.sks.data() + t * ks.sk_len);
    }
    return ks;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    const char* alg = oqs_alg_for(opt.tier);
    if (!alg) { std::fprintf(stderr, "unsupported tier %s\n", opt.tier.c_str()); return 2; }
    OQS_SIG* sig = OQS_SIG_new(alg);
    if (!sig) { std::fprintf(stderr, "liboqs does not provide %s\n", alg); return 2; }

    omp_set_num_threads(opt.threads);
    TenantKeyStore ks = build_keystore(sig, opt.tenants);

    // Synthetic per-request messages; tenant assignment is round-robin across the batch.
    std::vector<uint8_t> msgs(opt.batch * opt.msg_len);
    for (size_t i = 0; i < msgs.size(); ++i) msgs[i] = static_cast<uint8_t>(i * 2654435761u >> 24);
    std::vector<uint8_t> sigs(opt.batch * sig->length_signature);
    std::vector<size_t> siglens(opt.batch);

    std::vector<double> latencies_us(opt.batch);
    int verify_failures = 0;

    for (int trial = 0; trial < opt.trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(dynamic, 16)
        for (long i = 0; i < static_cast<long>(opt.batch); ++i) {
            size_t tenant = static_cast<size_t>(i) % ks.num_tenants;
            auto s = std::chrono::steady_clock::now();
            OQS_SIG_sign(sig, sigs.data() + i * sig->length_signature, &siglens[i],
                         msgs.data() + i * opt.msg_len, opt.msg_len, ks.sk(tenant));
            auto e = std::chrono::steady_clock::now();
            latencies_us[i] = std::chrono::duration<double, std::micro>(e - s).count();
        }
        auto t1 = std::chrono::steady_clock::now();

        double wall_s = std::chrono::duration<double>(t1 - t0).count();
        double throughput = static_cast<double>(opt.batch) / wall_s;
        std::vector<double> lat = latencies_us;
        double mean_us = std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();
        double p50 = percentile(lat, 0.50);
        double p99 = percentile(lat, 0.99);

        // Verify a sample of this trial's signatures against the matching tenant public key.
        size_t sample = std::min<size_t>(opt.batch, 256);
        for (size_t i = 0; i < sample; ++i) {
            size_t tenant = i % ks.num_tenants;
            if (OQS_SIG_verify(sig, msgs.data() + i * opt.msg_len, opt.msg_len,
                               sigs.data() + i * sig->length_signature, siglens[i],
                               ks.pk(tenant)) != OQS_SUCCESS) {
                ++verify_failures;
            }
        }

        if (opt.csv) {
            std::printf("cpu,%s,%zu,%d,%zu,%d,%.3f,%.3f,%.3f,%.3f,%d\n", opt.tier.c_str(), opt.batch,
                        opt.threads, opt.tenants, trial, throughput, mean_us, p50, p99, verify_failures);
        } else {
            std::printf("[cpu %s batch=%zu threads=%d trial=%d] %.1f sig/s  mean=%.1fus p50=%.1fus p99=%.1fus verify_fail=%d\n",
                        opt.tier.c_str(), opt.batch, opt.threads, trial, throughput, mean_us, p50, p99, verify_failures);
        }
        std::fflush(stdout);
    }

    OQS_SIG_free(sig);
    return verify_failures == 0 ? 0 : 1;
}
