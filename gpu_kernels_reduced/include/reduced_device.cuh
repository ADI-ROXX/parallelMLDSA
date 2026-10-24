// gpu_kernels_reduced/include/reduced_device.cuh
// ML-DSA-44 signing with 27 scratch polynomials per slot (see reduced_gpu.h).
// Signatures are byte-identical to mldsa_sign_device.cuh; only the scratch layout and the
// per-attempt A_hat regeneration differ.
#pragma once
#include "common_mldsa.cuh"
#include "mldsa_sign_device.cuh"
#include "reduced_gpu.h"

namespace mldsa {
namespace reduced {

constexpr int R_S1 = 0;    // 4 polys, NTT domain
constexpr int R_S2 = 4;    // 4 polys, NTT domain
constexpr int R_T0 = 8;    // 4 polys, NTT domain
constexpr int R_Y = 12;    // 4 polys, mask y (then z)
constexpr int R_YH = 16;   // 4 polys, NTT(y)
constexpr int R_W = 20;    // 4 polys, w (then w - c*s2)
constexpr int R_CH = 24;   // 1 poly, NTT(c)
constexpr int R_TMP = 25;  // 1 poly, c*s1 / c*s2 / c*t0
constexpr int R_A = 26;    // 1 poly, one entry of A_hat

__device__ inline void expand_a_entry(PolySlot a, ByteSlot rho, int r, int s) {
    Sponge g;
    sp_init(g, 168);
    sp_absorb(g, rho, 32);
    const uint8_t idx[2] = {static_cast<uint8_t>(s), static_cast<uint8_t>(r)};
    sp_absorb(g, idx, 2);
    sp_finalize(g);
    squeeze_uniform_poly(g, a);
}

__device__ inline void prepare_r_key(PolySlot S, ByteSlot B, const KeyView& kv, int tenant) {
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

    for (int l = 0; l < L; ++l)
        for (int n = 0; n < N; ++n) {
            int32_t v = ETA - static_cast<int32_t>(bits_at(pk_s1 + l * 96, n * 3, 3));
            S[(R_S1 + l) * N + n] = from_signed(v);
        }
    for (int i = 0; i < K; ++i)
        for (int n = 0; n < N; ++n) {
            int32_t v = ETA - static_cast<int32_t>(bits_at(pk_s2 + i * 96, n * 3, 3));
            S[(R_S2 + i) * N + n] = from_signed(v);
        }
    for (int i = 0; i < K; ++i)
        for (int n = 0; n < N; ++n) {
            int32_t v = 4096 - static_cast<int32_t>(bits_at(pk_t0 + i * 416, n * 13, 13));
            S[(R_T0 + i) * N + n] = from_signed(v);
        }
    for (int l = 0; l < L; ++l) ntt(S + (R_S1 + l) * N);
    for (int i = 0; i < K; ++i) ntt(S + (R_S2 + i) * N);
    for (int i = 0; i < K; ++i) ntt(S + (R_T0 + i) * N);
}

__device__ inline void prepare_r(PolySlot S, ByteSlot B, const KeyView& kv, int tenant, const uint8_t* msg) {
    prepare_r_key(S, B, kv, tenant);
    mldsa_prepare_msg(B, msg);
}

// One rejection-sampling iteration at counter kappa; writes sig only on acceptance.
__device__ inline bool attempt_r(PolySlot S, ByteSlot B, uint8_t* sig, uint32_t kappa) {
    for (int l = 0; l < L; ++l) {
        Sponge sp;
        sp_init(sp, 136);
        sp_absorb(sp, B + OFF_RHOPP, 64);
        const uint32_t ctr_val = kappa + l;
        const uint8_t ctr[2] = {static_cast<uint8_t>(ctr_val & 0xFF), static_cast<uint8_t>((ctr_val >> 8) & 0xFF)};
        sp_absorb(sp, ctr, 2);
        sp_finalize(sp);
        PolySlot y = S + (R_Y + l) * N;
        PolySlot yh = S + (R_YH + l) * N;
        expand_mask_poly(sp, y, yh);
        ntt(yh);
    }

    // w = InvNTT(A_hat * NTT(y)), with A_hat entries regenerated on use
    PolySlot a = S + R_A * N;
    for (int i = 0; i < K; ++i) {
        PolySlot w = S + (R_W + i) * N;
        for (int n = 0; n < N; ++n) w[n] = 0;
        for (int j = 0; j < L; ++j) {
            expand_a_entry(a, B + OFF_RHO, i, j);
            const PolySlot yh = S + (R_YH + j) * N;
            for (int n = 0; n < N; ++n) w[n] = addq(w[n], mulq(a[n], yh[n]));
        }
        invntt(w);
    }

    ByteSlot w1 = B + OFF_W1;
    for (int i = 0; i < W1_BYTES; ++i) w1[i] = 0;
    for (int i = 0; i < K; ++i)
        for (int n = 0; n < N; ++n) {
            uint32_t r1 = static_cast<uint32_t>(high_bits(S[(R_W + i) * N + n]));
            put_bits(w1, (i * N + n) * 6, r1, 6);
        }

    shake256_2(B + OFF_MU, 64, w1, W1_BYTES, B + OFF_CT, CTILDE_LEN);
    PolySlot ch = S + R_CH * N;
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

    PolySlot tmp = S + R_TMP * N;
    for (int l = 0; l < L; ++l) {
        const PolySlot s1h = S + (R_S1 + l) * N;
        for (int n = 0; n < N; ++n) tmp[n] = mulq(ch[n], s1h[n]);
        invntt(tmp);
        PolySlot y = S + (R_Y + l) * N;
        for (int n = 0; n < N; ++n) {
            y[n] = addq(y[n], tmp[n]);
            int32_t zc = centered(y[n]);
            if (zc >= GAMMA1 - BETA || zc <= -(GAMMA1 - BETA)) return false;
        }
    }

    // Overwrites w with u = w - c*s2 so the hint step can reuse it without storing c*s2.
    for (int i = 0; i < K; ++i) {
        const PolySlot s2h = S + (R_S2 + i) * N;
        for (int n = 0; n < N; ++n) tmp[n] = mulq(ch[n], s2h[n]);
        invntt(tmp);
        PolySlot w = S + (R_W + i) * N;
        for (int n = 0; n < N; ++n) {
            w[n] = subq(w[n], tmp[n]);
            int32_t r0 = low_bits_centered(w[n]);
            if (r0 >= GAMMA2 - BETA || r0 <= -(GAMMA2 - BETA)) return false;
        }
    }

    int hint_count = 0;
    for (int i = 0; i < K; ++i) {
        const PolySlot t0h = S + (R_T0 + i) * N;
        for (int n = 0; n < N; ++n) tmp[n] = mulq(ch[n], t0h[n]);
        invntt(tmp);
        const PolySlot u_row = S + (R_W + i) * N;
        for (int n = 0; n < N; ++n) {
            int32_t cc = centered(tmp[n]);
            if (cc >= GAMMA2 || cc <= -GAMMA2) return false;
            const uint32_t u = u_row[n];
            const uint32_t v = addq(u, tmp[n]);
            uint8_t h = high_bits(v) != high_bits(u) ? 1 : 0;
            B[OFF_H + i * N + n] = h;
            hint_count += h;
        }
    }
    if (hint_count > OMEGA) return false;

    for (int i = 0; i < SIG_LEN; ++i) sig[i] = 0;
    for (int i = 0; i < CTILDE_LEN; ++i) sig[i] = B[OFF_CT + i];
    constexpr int Z_OFF = CTILDE_LEN;
    constexpr int H_OFF = Z_OFF + L * N * 18 / 8;
    for (int l = 0; l < L; ++l) {
        const PolySlot z = S + (R_Y + l) * N;
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

}  // namespace reduced
}  // namespace mldsa