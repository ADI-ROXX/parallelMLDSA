# Next steps

Stop adding unmeasured micro-optimizations. Step 6 (packing `w1` four coefficients at a time and `z` four coefficients at a time) is a small change on a path that is not the bulk of an attempt. Do it only if a profile says packing shows up.

Nothing below has been run. The first job on a machine with `nvcc` is to find out whether `main` is correct.

## 1. Prove the current tree signs correctly

On the GPU machine:

```bash
make verify
```

That signs a batch with `naive`, `warpsign`, and `warpsign_reduced` and checks every signature with liboqs. If it fails, fix that before any paper numbers. The reciprocal `mulq` and the high/low-bits reduction matched a host spot check, and the streaming `ExpandMask` / `ExpandA` / `t0` unpackers have not been checked at all.

## 2. Fix the latency numbers

In both persistent kernels, stamp `t_start_ns` when the lane pops the request, not when the 8-lane prepare threshold finally lets it run `mldsa_prepare`. p50 and p99 otherwise hide the wait that threshold refill introduces. Revisit `warp_eff` in the same pass so the counter still means "useful lane-iterations / issued warp-iterations."

## 3. Decide which paper this is

These are different papers. Pick one before writing more kernels.

### A. Challenge cuML-DSA

The published server GPU designs are not one-thread-per-signature.

- Shen, Yang, Li, Zhao. *cuML-DSA: Optimized Signing Procedure and Server-Oriented GPU Design for ML-DSA*. IEEE TDSC, 2025. Open copy: https://eprint.iacr.org/2023/1522 . They assign 128 threads to one signature, and they treat `ExpandA` and secret-key unpacking as offline.
- Shen et al. *High-Throughput GPU Implementation of Dilithium Post-Quantum Digital Signature*. IEEE TPDS, 2024. Open copies: https://arxiv.org/abs/2211.12265 and https://eprint.iacr.org/2024/1365 . Code: https://github.com/encryptorion-lab/cuDilithium . One warp (32 threads) per signature, plus a 128-thread variant, batching, a memory pool, and multi-stream copies. About 575k Dilithium2 signatures/s on an A100 in the arXiv version.
- *ML-Cube* (CCS 2025) reports 30%–55% higher ML-DSA throughput than cuML-DSA using Tensor Cores, under the same server model.

To challenge them:

1. Build their code on the same GPU. If the repo will not build, reimplement the 128-thread-per-signature mapping and say that it is a reimplementation.
2. Match ML-DSA-44, their batch sizes, and their online/offline split.
3. Add a cached-tenant variant of WarpSign: expand `A_hat`, `s1`, `s2`, and `t0` once per tenant, and on a new message compute only `mu` and `rho''`. The current kernels redo that work on every request, which is closer to a fresh liboqs `sign` call and will lose a raw throughput table against cuML-DSA.
4. Keep the one-thread-per-request kernel as an ablation, not as the published baseline.

### B. Write the WarpSign / multi-tenant IoT paper

1. Freeze commit `4df5bf8` (tag it, or copy that naive kernel into `baseline/`) so the unoptimized one-thread-per-request result can be reproduced.
2. Report four rows: frozen naive, current naive (steps 1–5), current WarpSign, and liboqs on the CPU.
3. The scheduler claim is current naive versus current WarpSign. The "how far from where we started" claim is frozen naive versus current WarpSign. Quote cuML-DSA's published numbers only with the mapping difference stated (1 thread per request here, 32 or 128 threads per request there).

## 4. Then measure, and only then optimize

- `make run` for the batch sweep, and `make profile` for Nsight Compute warp efficiency and load coalescing.
- Use those counters to decide whether threshold refill, the Keccak unroll, or the reduced-scratch kernel is worth keeping.
- After the profile, the open implementation items are: multi-stream overlap of host/device copies, ML-DSA-65 and ML-DSA-87, and a real tenant-key cache if paper target A is chosen.

## Suggested order

1. `make verify`.
2. Fix the start timestamp.
3. Choose paper target A or B.
4. For A, bring up cuDilithium or cuML-DSA on the same GPU. For B, tag `4df5bf8` and add it as a benchmark row.
5. Run the sweep and the profiler before changing any more device code.
