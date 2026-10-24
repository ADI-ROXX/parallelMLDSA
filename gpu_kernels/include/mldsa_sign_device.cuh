// ML-DSA-44 signing (FIPS 204 Algorithm 7, deterministic rnd = 0) split into a per-request
// prepare step and a single rejection-sampling attempt, so schedulers can interleave attempts.
// gpu_kernels/include/mldsa_sign_device.cuh
#pragma once
#include "common_mldsa.cuh"

namespace mldsa {

// Key layout selector: AoS stores each key contiguously; SoA stores byte `off` of all tenants
// contiguously, so lanes reading the same offset for different tenants coalesce.
struct KeyView {
    const uint8_t* base;
    bool soa;
    int num_tenants;
    __device__ __forceinline__ uint8_t byte(int tenant, int off) const {
        return soa ? base[static_cast<size_t>(off) * num_tenants + tenant]
                   : base[static_cast<size_t>(tenant) * SK_LEN + off];
    }
};

__device__ __forceinline__ uint32_t bits_at(const uint8_t* buf, int bit_index, int nbits) {
    uint32_t v = 0;
    for (int b = 0; b < nbits; ++b) {
        int bi = bit_index + b;
        v |= static_cast<uint32_t>((buf[bi >> 3] >> (bi & 7)) & 1) << b;
    }
    return v;
}

__device__ __forceinline__ void put_bits(uint8_t* buf, int bit_index, uint32_t value, int nbits) {
    for (int b = 0; b < nbits; ++b) {
        int bi = bit_index + b;
        if ((value >> b) & 1u) buf[bi >> 3] |= static_cast<uint8_t>(1u << (bi & 7));
    }
}

// Coefficient of a high-bits decomposition (FIPS 204 Algorithm 37, HighBits).
__device__ __forceinline__ int32_t high_bits(uint32_t r) {
    int32_t rr = static_cast<int32_t>(r);
    int32_t r0 = rr % (2 * GAMMA2);
    if (r0 > GAMMA2) r0 -= 2 * GAMMA2;
    if (rr - r0 == static_cast<int32_t>(Q) - 1) return 0;
    return (rr - r0) / (2 * GAMMA2);
}

__device__ __forceinline__ int32_t low_bits_centered(uint32_t r) {
    int32_t r0 = static_cast<int32_t>(r) % (2 * GAMMA2);
    if (r0 > GAMMA2) r0 -= 2 * GAMMA2;
    return r0;
}

// Stage 0 of a request: unpack key, compute mu, rho'', A_hat and NTT-domain secrets.
__device__ inline void mldsa_prepare(uint32_t* S, uint8_t* B, const KeyView& kv, int tenant, const uint8_t* msg) {
    for (int i = 0; i < 32; ++i) {
        B[OFF_RHO + i] = kv.byte(tenant, i);
        B[OFF_K + i] = kv.byte(tenant, 32 + i);
    }
    for (int i = 0; i < 64; ++i) B[OFF_TR + i] = kv.byte(tenant, 64 + i);

    uint8_t pk_s1[L * 96];
    uint8_t pk_s2[K * 96];
    uint8_t pk_t0[K * 416];
    for (int i = 0; i < L * 96; ++i) pk_s1[i] = kv.byte(tenant, SK_OFF_S1 + i);
    for (int i = 0; i < K * 96; ++i) pk_s2[i] = kv.byte(tenant, SK_OFF_S2 + i);
    for (int i = 0; i < K * 416; ++i) pk_t0[i] = kv.byte(tenant, SK_OFF_T0 + i);

    // s1, s2: 3-bit fields, coefficient = ETA - x. t0: 13-bit fields, coefficient = 2^12 - x.
    for (int l = 0; l < L; ++l)
        for (int n = 0; n < N; ++n) {
            int32_t v = ETA - static_cast<int32_t>(bits_at(pk_s1 + l * 96, n * 3, 3));
            S[(P_S1 + l) * N + n] = from_signed(v);
        }
    for (int i = 0; i < K; ++i)
        for (int n = 0; n < N; ++n) {
            int32_t v = ETA - static_cast<int32_t>(bits_at(pk_s2 + i * 96, n * 3, 3));
            S[(P_S2 + i) * N + n] = from_signed(v);
        }
    for (int i = 0; i < K; ++i)
        for (int n = 0; n < N; ++n) {
            int32_t v = 4096 - static_cast<int32_t>(bits_at(pk_t0 + i * 416, n * 13, 13));
            S[(P_T0 + i) * N + n] = from_signed(v);
        }
    for (int l = 0; l < L; ++l) ntt(S + (P_S1 + l) * N);
    for (int i = 0; i < K; ++i) ntt(S + (P_S2 + i) * N);
    for (int i = 0; i < K; ++i) ntt(S + (P_T0 + i) * N);

    // mu = H(tr || 0x00 || 0x00 || msg, 64), empty context (FIPS 204 Algorithm 2)
    const uint8_t prefix[2] = {0, 0};
    Sponge sp;
    sp_init(sp, 136);
    sp_absorb(sp, B + OFF_TR, 64);
    sp_absorb(sp, prefix, 2);
    sp_absorb(sp, msg, MSG_LEN);
    sp_finalize(sp);
    sp_squeeze(sp, B + OFF_MU, 64);

    // rho'' = H(K || rnd || mu, 64), rnd = 0^32 (deterministic signing)
    const uint8_t rnd[32] = {0};
    Sponge sp2;
    sp_init(sp2, 136);
    sp_absorb(sp2, B + OFF_K, 32);
    sp_absorb(sp2, rnd, 32);
    sp_absorb(sp2, B + OFF_MU, 64);
    sp_finalize(sp2);
    sp_squeeze(sp2, B + OFF_RHOPP, 64);

    // A_hat = ExpandA(rho), sampled directly in the NTT domain (FIPS 204 Algorithm 32).
    for (int r = 0; r < K; ++r)
        for (int s = 0; s < L; ++s) {
            uint32_t* a = S + (P_AH + r * L + s) * N;
            Sponge g;
            sp_init(g, 168);
            sp_absorb(g, B + OFF_RHO, 32);
            const uint8_t idx[2] = {static_cast<uint8_t>(s), static_cast<uint8_t>(r)};
            sp_absorb(g, idx, 2);
            sp_finalize(g);
            int count = 0;
            while (count < N) {
                uint8_t t[3];
                sp_squeeze(g, t, 3);
                uint32_t v = t[0] | (static_cast<uint32_t>(t[1]) << 8) | (static_cast<uint32_t>(t[2] & 0x7F) << 16);
                if (v < Q) a[count++] = v;
            }
        }
}

// One rejection-sampling iteration (FIPS 204 Algorithm 7, steps 5-29) at counter kappa.
// Returns true and writes the signature to `sig` on acceptance.
__device__ inline bool mldsa_attempt(uint32_t* S, uint8_t* B, uint8_t* sig, uint32_t kappa) {
    // y = ExpandMask(rho'', kappa), 18-bit fields, coefficient = GAMMA1 - x
    for (int l = 0; l < L; ++l) {
        uint8_t stream[576];
        Sponge sp;
        sp_init(sp, 136);
        sp_absorb(sp, B + OFF_RHOPP, 64);
        const uint32_t ctr_val = kappa + l;
        const uint8_t ctr[2] = {static_cast<uint8_t>(ctr_val & 0xFF), static_cast<uint8_t>((ctr_val >> 8) & 0xFF)};
        sp_absorb(sp, ctr, 2);
        sp_finalize(sp);
        sp_squeeze(sp, stream, 576);
        uint32_t* y = S + (P_Y + l) * N;
        uint32_t* yh = S + (P_YH + l) * N;
        for (int n = 0; n < N; ++n) {
            int32_t x = static_cast<int32_t>(bits_at(stream, n * 18, 18));
            y[n] = from_signed(GAMMA1 - x);
            yh[n] = y[n];
        }
        ntt(yh);
    }

    // w = InvNTT(A_hat * NTT(y))
    for (int i = 0; i < K; ++i) {
        uint32_t* w = S + (P_W + i) * N;
        for (int n = 0; n < N; ++n) {
            uint32_t acc = 0;
            for (int j = 0; j < L; ++j)
                acc = addq(acc, mulq(S[(P_AH + i * L + j) * N + n], S[(P_YH + j) * N + n]));
            w[n] = acc;
        }
        invntt(w);
    }

    // w1Encode(HighBits(w)), 6 bits per coefficient
    uint8_t* w1 = B + OFF_W1;
    for (int i = 0; i < W1_BYTES; ++i) w1[i] = 0;
    for (int i = 0; i < K; ++i)
        for (int n = 0; n < N; ++n) {
            uint32_t r1 = static_cast<uint32_t>(high_bits(S[(P_W + i) * N + n]));
            put_bits(w1, (i * N + n) * 6, r1, 6);
        }

    // c_tilde = H(mu || w1Encode(w1), 32); c = SampleInBall(c_tilde); c_hat = NTT(c)
    shake256_2(B + OFF_MU, 64, w1, W1_BYTES, B + OFF_CT, CTILDE_LEN);
    uint32_t* ch = S + P_CH * N;
    for (int n = 0; n < N; ++n) ch[n] = 0;
    {
        Sponge sp;
        sp_init(sp, 136);
        sp_absorb(sp, B + OFF_CT, CTILDE_LEN);
        sp_finalize(sp);
        uint8_t sb[8];
        sp_squeeze(sp, sb, 8);
        uint64_t signs = 0;
        for (int i = 0; i < 8; ++i) signs |= static_cast<uint64_t>(sb[i]) << (8 * i);
        for (int i = N - TAU; i < N; ++i) {
            uint8_t jb;
            do {
                sp_squeeze(sp, &jb, 1);
            } while (jb > i);
            const int j = jb;
            ch[i] = ch[j];
            ch[j] = (signs & 1ull) ? Q - 1 : 1u;
            signs >>= 1;
        }
    }
    ntt(ch);

    // z = y + c*s1, with ||z||inf < GAMMA1 - BETA
    for (int l = 0; l < L; ++l) {
        uint32_t* cs1 = S + (P_CS1 + l) * N;
        uint32_t* s1h = S + (P_S1 + l) * N;
        for (int n = 0; n < N; ++n) cs1[n] = mulq(ch[n], s1h[n]);
        invntt(cs1);
        uint32_t* y = S + (P_Y + l) * N;
        for (int n = 0; n < N; ++n) {
            y[n] = addq(y[n], cs1[n]);
            int32_t zc = centered(y[n]);
            if (zc >= GAMMA1 - BETA || zc <= -(GAMMA1 - BETA)) return false;
        }
    }

    // ||LowBits(w - c*s2)||inf < GAMMA2 - BETA
    for (int i = 0; i < K; ++i) {
        uint32_t* cs2 = S + (P_CS2 + i) * N;
        uint32_t* s2h = S + (P_S2 + i) * N;
        for (int n = 0; n < N; ++n) cs2[n] = mulq(ch[n], s2h[n]);
        invntt(cs2);
        uint32_t* w = S + (P_W + i) * N;
        for (int n = 0; n < N; ++n) {
            int32_t r0 = low_bits_centered(subq(w[n], cs2[n]));
            if (r0 >= GAMMA2 - BETA || r0 <= -(GAMMA2 - BETA)) return false;
        }
    }

    // ||c*t0||inf < GAMMA2, then hints h = MakeHint(-ct0, w - cs2 + ct0) stored in B[OFF_H..]
    uint32_t* ct0 = S + P_TMP * N;
    int hint_count = 0;
    for (int i = 0; i < K; ++i) {
        uint32_t* t0h = S + (P_T0 + i) * N;
        for (int n = 0; n < N; ++n) ct0[n] = mulq(ch[n], t0h[n]);
        invntt(ct0);
        uint32_t* w = S + (P_W + i) * N;
        uint32_t* cs2 = S + (P_CS2 + i) * N;
        for (int n = 0; n < N; ++n) {
            int32_t cc = centered(ct0[n]);
            if (cc >= GAMMA2 || cc <= -GAMMA2) return false;
            uint32_t u = subq(w[n], cs2[n]);
            uint32_t v = addq(u, ct0[n]);
            uint8_t h = high_bits(v) != high_bits(u) ? 1 : 0;
            B[OFF_H + i * N + n] = h;
            hint_count += h;
        }
    }
    if (hint_count > OMEGA) return false;

    // Signature: c_tilde (32) || z packed 18 bits per coefficient (2304) || hint encoding (80 + 4)
    for (int i = 0; i < SIG_LEN; ++i) sig[i] = 0;
    for (int i = 0; i < CTILDE_LEN; ++i) sig[i] = B[OFF_CT + i];
    constexpr int Z_OFF = CTILDE_LEN;
    constexpr int H_OFF = Z_OFF + L * N * 18 / 8;
    for (int l = 0; l < L; ++l) {
        const uint32_t* z = S + (P_Y + l) * N;
        for (int n = 0; n < N; ++n) {
            int32_t v = GAMMA1 - centered(z[n]);
            put_bits(sig + Z_OFF, (l * N + n) * 18, static_cast<uint32_t>(v), 18);
        }
    }
    int idx = 0;
    for (int i = 0; i < K; ++i) {
        for (int n = 0; n < N; ++n)
            if (B[OFF_H + i * N + n]) sig[H_OFF + idx++] = static_cast<uint8_t>(n);
        sig[H_OFF + OMEGA + i] = static_cast<uint8_t>(idx);
    }
    return true;
}

}  // namespace mldsa