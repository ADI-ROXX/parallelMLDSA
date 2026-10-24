# What was done

This note records the GPU ML-DSA-44 changes made on `main` after the starting commit `4df5bf8`. The goal of the code is a server that signs attestations for many IoT devices. Every change below is meant to keep FIPS 204 ML-DSA-44 signatures byte-compatible with liboqs. None of these commits has been compiled or checked with `make verify` yet. There is no CUDA toolkit on the machine that produced them.

## Starting point (`4df5bf8`)

- ML-DSA-44 only (`K = L = 4`, `Q = 8380417`).
- Device signing split into `mldsa_prepare` and one rejection attempt, `mldsa_attempt`.
- Three kernels:
  - `naive`: one thread per request, each thread runs its own rejection loop.
  - `warpsign`: persistent warps. A lane that accepts immediately pops the next request and prepares it, while the other lanes in the warp wait.
  - `warpsign_reduced`: same scheduler, 27 polynomials per slot, `A_hat` regenerated on each attempt.
- Secret keys could already be stored as an array of structures or as a structure of arrays. The per-thread scratchpad was an array of structures: slot `i` owned a contiguous block of about 51 KB, so a warp reading one coefficient issued 32 loads about 51 KB apart.
- Arithmetic used 64-bit `% Q`. Keccak, bit packing, `ExpandA`, `ExpandMask`, and secret-key unpacking were byte- or bit-at-a-time, with large per-thread stack arrays.
- Host benchmark against multi-threaded liboqs, plus a liboqs verification gate.

## Step 1 — warp-striped scratchpad

Commit `ff1c65b` (shared with steps 2's arithmetic and the first refill change).

- Added `WarpSlice` and `warp_slot` in `gpu_kernels/include/common_mldsa.cuh`.
- Inside a warp, element `i` of lane `l` lives at `base[i * 32 + l]`. All 32 lanes reading the same coefficient or byte offset now touch consecutive addresses.
- `ntt`, `invntt`, the sponge helpers, `mldsa_prepare`, `mldsa_attempt`, `prepare_r`, and `attempt_r` take these views instead of raw pointers.
- Naive slot count is rounded up to a multiple of 32 so a partial warp still has a full stripe.
- Applies to `naive`, `warpsign`, and `warpsign_reduced`.

## Step 2 — faster modular arithmetic and threshold refill

Same commit `ff1c65b` for the arithmetic. The refill logic is in `gpu_kernels/src/warpsign_compact_kernel.cu` and `gpu_kernels_reduced/src/warpsign_reduced_kernel.cu`.

Arithmetic:

- `mulq` uses a reciprocal multiply (`floor(2^46 / Q) = 8396807`) instead of a 64-bit remainder. A host check of 200 million random and edge inputs found 0 mismatches. That is evidence, not a proof for every pair.
- `high_bits` and `low_bits_centered` divide by `2 * GAMMA2 = 190464` with reciprocal `22550`. Every residue in `[0, Q)` matched the original formula on the host.
- `from_signed` no longer uses `% Q`.
- Inner Keccak loops are marked `#pragma unroll`. That can raise register pressure and lower occupancy. It has not been measured.

Refill:

- `mldsa_prepare` is split into `mldsa_prepare_key` (unpack `s1`, `s2`, `t0`, 12 NTTs, `ExpandA`) and `mldsa_prepare_msg` (`mu` and `rho''`). `prepare_r` has the same split. Each request still runs both, from the raw 2560-byte secret key.
- A lane that accepts records its end time, pops the next request, and sets `pending_prep`. Preparation runs when at least 8 lanes in the warp are waiting, or when no lane is left in the attempt stage.
- This changes the reported latency. `t_start_ns` is stamped when preparation begins, so time spent waiting for the 8-lane threshold is missing from p50 and p99. `warp_eff` counts a different thing than it did before the change.

`OPTION2_SHELVED.md` describes a stronger version of this idea (a separate prepare wave, or caching tenant keys across requests). That stronger version is not in the kernels.

## Step 3 — word-level sponge and streaming ExpandMask

Commit `8309dac`.

- `sp_absorb` and `sp_squeeze` move 8 bytes at a time when the sponge position is 8-byte aligned, and fall back to one byte otherwise.
- `expand_mask_poly` reads 64-bit Keccak words straight into `y` and `NTT(y)`. The 576-byte `stream[]` stack buffer is gone from `mldsa_attempt` and `attempt_r`.

## Step 4 — streaming ExpandA

Commit `4a331fa`.

- `squeeze_uniform_poly` pulls 23-bit candidates from the SHAKE128 state in registers.
- Used by `mldsa_prepare_key` (16 polynomials, once per request) and by `expand_a_entry` (16 polynomials on every attempt of the reduced kernel).
- Replaces the old loop that called `sp_squeeze` for 3 bytes per candidate.

## Step 5 — unpack the secret key in registers

Commit `1b956fe`.

- `unpack_poly_eta` unpacks 3-bit `s1` and `s2` coefficients, 8 at a time, directly from `KeyView`.
- `unpack_poly_t0` unpacks 13-bit `t0` coefficients, 8 at a time.
- Removed the per-thread stack copies `pk_s1[384]`, `pk_s2[384]`, and `pk_t0[1664]` (2432 bytes) from both prepare paths. Each polynomial is NTT'd as soon as it is unpacked.

## What was not done

- No `nvcc` build, no `make verify`, no benchmark numbers.
- No Montgomery reduction. The reciprocal `mulq` is the version that landed.
- No multi-stream PCIe overlap, no ML-DSA-65, no ML-DSA-87.
- No frozen copy of the pre-optimization naive kernel. It still exists at commit `4df5bf8`.
- No comparison against published GPU code.

## Expected effect, not measured

Steps 1 and the arithmetic in step 2 should matter most: the scratchpad loads were badly strided, and `% Q` sat on every NTT butterfly. Steps 3–5 remove per-byte Keccak overhead and a 2.4 KB local-memory copy per request. The threshold refill can help or hurt, because lanes now wait before the expensive prepare. Steps 1 and 3–5 also speed up the naive kernel, so a WarpSign-versus-naive ratio on the new code does not isolate the scheduler.
