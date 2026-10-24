// ML-DSA-44 (FIPS 204) device primitives shared by all GPU kernels.
// gpu_kernels/include/common_mldsa.cuh

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace mldsa {

constexpr uint32_t Q = 8380417u;
constexpr int N = 256;
constexpr int K = 4;
constexpr int L = 4;
constexpr int ETA = 2;
constexpr int TAU = 39;
constexpr int BETA = TAU * ETA;
constexpr int32_t GAMMA1 = 1 << 17;
constexpr int32_t GAMMA2 = 95232;
constexpr int OMEGA = 80;
constexpr int CTILDE_LEN = 32;
constexpr int PK_LEN = 1312;
constexpr int SK_LEN = 2560;
constexpr int SIG_LEN = 2420;
constexpr int MSG_LEN = 64;
constexpr uint32_t MAX_ATTEMPTS = 1000;

// SK byte offsets (FIPS 204 skEncode): rho | K | tr | s1 | s2 | t0
constexpr int SK_OFF_S1 = 128;
constexpr int SK_OFF_S2 = SK_OFF_S1 + L * 96;
constexpr int SK_OFF_T0 = SK_OFF_S2 + K * 96;

// Per-slot scratch: polynomial indices (each polynomial is 256 uint32 words).
constexpr int P_AH = 0;    // 16 polys: A_hat[i*L+j]
constexpr int P_S1 = 16;   // 4 polys, NTT domain
constexpr int P_S2 = 20;   // 4 polys, NTT domain
constexpr int P_T0 = 24;   // 4 polys, NTT domain
constexpr int P_Y = 28;    // 4 polys, mask y (then z)
constexpr int P_YH = 32;   // 4 polys, NTT(y)
constexpr int P_W = 36;    // 4 polys, w
constexpr int P_CS1 = 40;  // 4 polys, c*s1
constexpr int P_CS2 = 44;  // 4 polys, c*s2
constexpr int P_CH = 48;   // 1 poly, NTT(c)
constexpr int P_TMP = 49;  // 1 poly, scratch
constexpr int SCRATCH_POLYS = 50;
constexpr int SCRATCH_WORDS = SCRATCH_POLYS * N;

// Per-slot byte scratch.
constexpr int OFF_RHO = 0;
constexpr int OFF_K = 32;
constexpr int OFF_TR = 64;
constexpr int OFF_MU = 128;
constexpr int OFF_RHOPP = 192;
constexpr int OFF_CT = 256;
constexpr int OFF_W1 = 288;
constexpr int W1_BYTES = K * N * 6 / 8;  // 768
constexpr int OFF_H = OFF_W1 + W1_BYTES;
constexpr int BYTES_PER_SLOT = OFF_H + K * N;

// Zeta powers in bit-reversed order (FIPS 204 Appendix B), filled by upload_tables().
__constant__ uint32_t c_zetas[N];

__device__ __forceinline__ uint32_t mulq(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % Q);
}
__device__ __forceinline__ uint32_t addq(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= Q ? s - Q : s;
}
__device__ __forceinline__ uint32_t subq(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + Q - b;
}
// Centered representative in (-(Q-1)/2, (Q-1)/2].
__device__ __forceinline__ int32_t centered(uint32_t a) {
    return a > (Q - 1) / 2 ? static_cast<int32_t>(a) - static_cast<int32_t>(Q) : static_cast<int32_t>(a);
}
__device__ __forceinline__ uint32_t from_signed(int32_t v) {
    int32_t r = v % static_cast<int32_t>(Q);
    return static_cast<uint32_t>(r < 0 ? r + static_cast<int32_t>(Q) : r);
}

// Keccak-f[1600] and a byte-oriented sponge for SHAKE128/SHAKE256.
__device__ inline void keccak_f(uint64_t* a) {
    constexpr uint64_t RC[24] = {
        0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808Aull, 0x8000000080008000ull,
        0x000000000000808Bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
        0x000000000000008Aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000Aull,
        0x000000008000808Bull, 0x800000000000008Bull, 0x8000000000008089ull, 0x8000000000008003ull,
        0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800Aull, 0x800000008000000Aull,
        0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};
    constexpr int ROT[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43,
                             25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
    for (int round = 0; round < 24; ++round) {
        uint64_t c[5], d[5];
        for (int x = 0; x < 5; ++x) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; ++x) {
            uint64_t r = (c[(x + 1) % 5] << 1) | (c[(x + 1) % 5] >> 63);
            d[x] = c[(x + 4) % 5] ^ r;
        }
        for (int i = 0; i < 25; ++i) a[i] ^= d[i % 5];
        uint64_t b[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y) {
                uint64_t v = a[x + 5 * y];
                int r = ROT[x + 5 * y];
                uint64_t rv = r == 0 ? v : ((v << r) | (v >> (64 - r)));
                b[y + 5 * ((2 * x + 3 * y) % 5)] = rv;
            }
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 5; ++x)
                a[x + 5 * y] = b[x + 5 * y] ^ (~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
        a[0] ^= RC[round];
    }
}

struct Sponge {
    uint64_t s[25];
    int rate;
    int pos;
};

__device__ inline void sp_init(Sponge& sp, int rate) {
    for (int i = 0; i < 25; ++i) sp.s[i] = 0;
    sp.rate = rate;
    sp.pos = 0;
}

__device__ inline void sp_absorb(Sponge& sp, const uint8_t* in, int len) {
    for (int i = 0; i < len; ++i) {
        sp.s[sp.pos >> 3] ^= static_cast<uint64_t>(in[i]) << (8 * (sp.pos & 7));
        if (++sp.pos == sp.rate) {
            keccak_f(sp.s);
            sp.pos = 0;
        }
    }
}

// Applies SHAKE padding and switches the sponge to squeezing.
__device__ inline void sp_finalize(Sponge& sp) {
    sp.s[sp.pos >> 3] ^= static_cast<uint64_t>(0x1F) << (8 * (sp.pos & 7));
    sp.s[(sp.rate - 1) >> 3] ^= static_cast<uint64_t>(0x80) << (8 * ((sp.rate - 1) & 7));
    keccak_f(sp.s);
    sp.pos = 0;
}

__device__ inline void sp_squeeze(Sponge& sp, uint8_t* out, int len) {
    for (int i = 0; i < len; ++i) {
        if (sp.pos == sp.rate) {
            keccak_f(sp.s);
            sp.pos = 0;
        }
        out[i] = static_cast<uint8_t>(sp.s[sp.pos >> 3] >> (8 * (sp.pos & 7)));
        ++sp.pos;
    }
}

__device__ inline void shake256(const uint8_t* in, int len, uint8_t* out, int outlen) {
    Sponge sp;
    sp_init(sp, 136);
    sp_absorb(sp, in, len);
    sp_finalize(sp);
    sp_squeeze(sp, out, outlen);
}

// Two-segment input, used to hash (prefix || suffix) without copying.
__device__ inline void shake256_2(const uint8_t* a, int alen, const uint8_t* b, int blen, uint8_t* out, int outlen) {
    Sponge sp;
    sp_init(sp, 136);
    sp_absorb(sp, a, alen);
    sp_absorb(sp, b, blen);
    sp_finalize(sp);
    sp_squeeze(sp, out, outlen);
}

// In-place forward NTT (FIPS 204 Algorithm 41).
__device__ inline void ntt(uint32_t* a) {
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1)
        for (int start = 0; start < N; start += 2 * len) {
            uint32_t zeta = c_zetas[++k];
            for (int j = start; j < start + len; ++j) {
                uint32_t t = mulq(zeta, a[j + len]);
                a[j + len] = subq(a[j], t);
                a[j] = addq(a[j], t);
            }
        }
}

// In-place inverse NTT (FIPS 204 Algorithm 42), including the 1/256 scaling.
__device__ inline void invntt(uint32_t* a) {
    constexpr uint32_t INV_N = 8347681u;  // 256^{-1} mod Q
    int k = N;
    for (int len = 1; len < N; len <<= 1)
        for (int start = 0; start < N; start += 2 * len) {
            uint32_t zeta = Q - c_zetas[--k];
            for (int j = start; j < start + len; ++j) {
                uint32_t t = a[j];
                a[j] = addq(t, a[j + len]);
                a[j + len] = mulq(zeta, subq(t, a[j + len]));
            }
        }
    for (int j = 0; j < N; ++j) a[j] = mulq(a[j], INV_N);
}

// Host-side table generation; each translation unit that uses the device primitives calls this once.
inline void upload_tables() {
    uint32_t host[N];
    for (int k = 0; k < N; ++k) {
        int brv = 0;
        for (int b = 0; b < 8; ++b) brv |= ((k >> b) & 1) << (7 - b);
        uint64_t z = 1, base = 1753;
        for (int e = 0; e < brv; ++e) z = (z * base) % Q;
        host[k] = static_cast<uint32_t>(z);
    }
    cudaMemcpyToSymbol(c_zetas, host, sizeof(host));
}

}  // namespace mldsa