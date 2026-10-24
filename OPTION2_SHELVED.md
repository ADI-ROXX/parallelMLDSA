# Shelved Optimization: Option 2 (Decoupling `mldsa_prepare` from Rejection-Sampling Loop)

**Status:** On the shelf (to be used as an alternative or extension to Option 1 if needed).

## Problem Addressed
In `warpsign_compact_kernel.cu` and `warpsign_reduced_kernel.cu`, each warp executes rejection-sampling attempts in lockstep across 32 lanes. For ML-DSA-44, the per-attempt acceptance probability is roughly $P_{\text{accept}} \approx 0.235$, meaning on average ~7.5 lanes in a 32-lane warp accept per iteration while ~24.5 lanes reject.

Currently, inside `if (active && accepted)`, the ~7.5 accepted lanes immediately pull a new request from the atomic queue and run `mldsa_prepare` (or `prepare_r`), which performs:
1. Key unpacking of $s_1, s_2, t_0$
2. 12 forward NTTs ($\hat{s}_1, \hat{s}_2, \hat{t}_0$)
3. Two SHAKE-256 hashes ($\mu$ and $\rho''$)
4. 16 SHAKE-128 sponges for `ExpandA` ($\hat{A}$, in the full-scratch kernel)

While those ~7.5 lanes execute `mldsa_prepare`, the remaining ~24.5 active lanes that rejected in that iteration sit idle waiting for the divergent branch to reconverge at `__syncwarp(FULL)`.

## How Option 2 Works (When Activated)
Instead of running full `mldsa_prepare` divergently inside the `if (active && accepted)` branch:

### Variant 2A: Two-Stage Batch Pipeline (Strict Per-Request Raw SK Equivalence)
- **Stage 1 (Preparation Kernel / Wave):** All 32 lanes in a warp cooperatively run `mldsa_prepare` for a batch of requests in lockstep (100% warp efficiency, zero branch divergence on the 12 NTTs and `ExpandA`).
- **Stage 2 (Persistent Rejection-Sampling Kernel):** Lanes pop already-prepared slots and run only `mldsa_attempt` until acceptance, eliminating the heavy refill stall from the rejection loop.

### Variant 2B: Multi-Tenant IoT Key Pre-Expansion
- Because an IoT attestation server signs many messages using a fixed set of `num_tenants` secret keys, the tenant-dependent state ($\hat{A}, \hat{s}_1, \hat{s}_2, \hat{t}_0$) can be expanded once per tenant on the GPU.
- When a lane finishes a signature and pulls a new request `(req, tenant, msg)` from `queue`, the refill step inside `if (active && accepted)` only computes:
  - $\mu = \text{SHAKE256}(tr \parallel 0 \parallel 0 \parallel \text{msg})$
  - $\rho'' = \text{SHAKE256}(K \parallel \text{rnd} \parallel \mu)$
- This shrinks per-slot scratch from 50 polynomials (200 bytes/poly $\times$ 50 = 51.2 KB) down to 14 polynomials (~14.3 KB) and reduces refill divergence cost by >15x.
