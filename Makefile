# WarpSign-PQC benchmark harness (ML-DSA-44)
# Requirements: CUDA toolkit (nvcc, sm_80/sm_90), GCC 11+ with OpenMP, liboqs under $(LIBOQS_PREFIX),
#               Python 3 (standard library only) for metric scoring, Nsight Compute (ncu) for `make profile`.
#
#   make verify    sign on GPU with every kernel and check each signature with liboqs (correctness gate)
#   make run       verify -> CPU sweep -> GPU sweep (naive, warpsign) -> metrics
#   make run1      verify -> GPU sweep with batches that exceed the full-device slot count (queue refill)
#   make run2      verify -> GPU sweep with naive, warpsign and the reduced-scratch kernel (large batches)
#   make metrics   regenerate results/metrics.md from the CSVs
#   make profile   ncu warp-efficiency and coalescing counters for the paper
#   make clean

LIBOQS_PREFIX ?= /usr/local
NVCC          ?= nvcc
CXX           ?= g++
GPU_ARCHS     ?= 80 90
BATCHES       ?= 256 1024 4096 16384 65536
RUN1_BATCHES  ?= 262144 524288 1048576
RUN2_BATCHES  ?= 262144 524288 1048576
TIER          ?= mldsa44
TRIALS        ?= 5
TENANTS       ?= 1024
CPU_THREADS   ?= 32
GPU_IDS       ?= 0
PROFILE_BATCH ?= 2048
PYTHON        ?= python3

BIN     := bin
RESULTS := results

CXXFLAGS := -O3 -std=c++17 -fopenmp -march=native -Wall -I$(LIBOQS_PREFIX)/include
LDFLAGS  := -L$(LIBOQS_PREFIX)/lib -Wl,-rpath,$(LIBOQS_PREFIX)/lib -loqs -fopenmp
NVFLAGS  := -O3 -std=c++17 $(foreach a,$(GPU_ARCHS),-gencode arch=compute_$(a),code=sm_$(a)) \
            -Igpu_kernels/include -Igpu_kernels_reduced/include -I$(LIBOQS_PREFIX)/include
NVLIBS   := -L$(LIBOQS_PREFIX)/lib -Xlinker -rpath,$(LIBOQS_PREFIX)/lib -loqs

CPU_BENCH  := $(BIN)/cpu_mldsa_bench
GPU_BENCH  := $(BIN)/gpu_mldsa_bench
GPU_VERIFY := $(BIN)/gpu_mldsa_verify

GPU_HDRS := $(wildcard gpu_kernels/include/*.cuh gpu_kernels/include/*.h gpu_kernels_reduced/include/*.cuh gpu_kernels_reduced/include/*.h)
GPU_SRCS := gpu_kernels/src/naive_batch_kernel.cu gpu_kernels/src/warpsign_compact_kernel.cu \
            gpu_kernels_reduced/src/warpsign_reduced_kernel.cu

CPU_HEADER := impl,tier,batch,threads,tenants,trial,throughput_sig_s,mean_us,p50_us,p99_us,verify_fail
GPU_HEADER := impl,kernel,gpu,tier,batch,tenants,trial,kernel_ms,throughput_sig_s,mean_us,p50_us,p99_us,mean_attempts,warp_eff,failures,layout

.PHONY: all verify run run1 run2 run-cpu run-gpu metrics profile clean

all: $(CPU_BENCH) $(GPU_BENCH) $(GPU_VERIFY)

$(BIN):
	mkdir -p $@

$(RESULTS):
	mkdir -p $@

$(CPU_BENCH): cpu_baseline/cpu_mldsa_bench.cpp cpu_baseline/db_mock_fetcher.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) cpu_baseline/cpu_mldsa_bench.cpp -o $@ $(LDFLAGS)

$(GPU_BENCH): benchmarks/gpu_mldsa_bench.cu $(GPU_SRCS) $(GPU_HDRS) | $(BIN)
	$(NVCC) $(NVFLAGS) benchmarks/gpu_mldsa_bench.cu $(GPU_SRCS) -o $@ $(NVLIBS)

$(GPU_VERIFY): benchmarks/gpu_mldsa_verify.cu $(GPU_SRCS) $(GPU_HDRS) | $(BIN)
	$(NVCC) $(NVFLAGS) benchmarks/gpu_mldsa_verify.cu $(GPU_SRCS) -o $@ $(NVLIBS)

verify: $(GPU_VERIFY) | $(RESULTS)
	$(GPU_VERIFY) --tier $(TIER) --gpu $(firstword $(GPU_IDS)) --batch 2048 --tenants $(TENANTS) > $(RESULTS)/verify.txt
	cat $(RESULTS)/verify.txt

run: all verify run-cpu run-gpu metrics

run1: all verify | $(RESULTS)
	@echo "$(GPU_HEADER)" > $(RESULTS)/gpu_run1.csv
	@for g in $(GPU_IDS); do \
	  for k in naive warpsign; do \
	    for b in $(RUN1_BATCHES); do \
	      $(GPU_BENCH) --kernel $$k --gpu $$g --batch $$b --tier $(TIER) --trials $(TRIALS) \
	        --tenants $(TENANTS) --csv >> $(RESULTS)/gpu_run1.csv || exit 1; \
	    done; \
	  done; \
	done
	cat $(RESULTS)/gpu_run1.csv

run2: all verify | $(RESULTS)
	@echo "$(GPU_HEADER)" > $(RESULTS)/gpu_run2.csv
	@for g in $(GPU_IDS); do \
	  for k in naive warpsign warpsign_reduced; do \
	    for b in $(RUN2_BATCHES); do \
	      $(GPU_BENCH) --kernel $$k --gpu $$g --batch $$b --tier $(TIER) --trials $(TRIALS) \
	        --tenants $(TENANTS) --csv >> $(RESULTS)/gpu_run2.csv || exit 1; \
	    done; \
	  done; \
	done
	cat $(RESULTS)/gpu_run2.csv

run-cpu: $(CPU_BENCH) | $(RESULTS)
	@[ -f $(RESULTS)/cpu.csv ] || echo "$(CPU_HEADER)" > $(RESULTS)/cpu.csv
	@for b in $(BATCHES); do \
	  $(CPU_BENCH) --threads $(CPU_THREADS) --batch $$b --tier $(TIER) --trials $(TRIALS) \
	    --tenants $(TENANTS) --csv >> $(RESULTS)/cpu.csv || exit 1; \
	done

run-gpu: $(GPU_BENCH) | $(RESULTS)
	@[ -f $(RESULTS)/gpu.csv ] || echo "$(GPU_HEADER)" > $(RESULTS)/gpu.csv
	@for g in $(GPU_IDS); do \
	  for k in naive warpsign; do \
	    for b in $(BATCHES); do \
	      $(GPU_BENCH) --kernel $$k --gpu $$g --batch $$b --tier $(TIER) --trials $(TRIALS) \
	        --tenants $(TENANTS) --csv >> $(RESULTS)/gpu.csv || exit 1; \
	    done; \
	  done; \
	done

metrics: | $(RESULTS)
	$(PYTHON) benchmarks/score_results.py --cpu $(RESULTS)/cpu.csv --gpu $(RESULTS)/gpu.csv \
	  --out $(RESULTS)/metrics.md

# Nsight Compute counters: warp efficiency, divergence stalls, and global load coalescing.
profile: $(GPU_BENCH) | $(RESULTS)
	ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio,sm__warp_issue_stalled_barrier_pct \
	    $(GPU_BENCH) --kernel warpsign --gpu $(firstword $(GPU_IDS)) --batch $(PROFILE_BATCH) --trials 1 \
	    > $(RESULTS)/ncu_warp_efficiency.txt 2>&1
	ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.ratio \
	    $(GPU_BENCH) --kernel warpsign --gpu $(firstword $(GPU_IDS)) --batch $(PROFILE_BATCH) --trials 1 \
	    > $(RESULTS)/ncu_coalescing.txt 2>&1
	ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio \
	    $(GPU_BENCH) --kernel naive --gpu $(firstword $(GPU_IDS)) --batch $(PROFILE_BATCH) --trials 1 \
	    > $(RESULTS)/ncu_naive_warp_efficiency.txt 2>&1

clean:
	rm -rf $(BIN) $(RESULTS)