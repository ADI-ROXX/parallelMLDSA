#!/usr/bin/env python3
"""Aggregate CPU/GPU benchmark CSVs into paper-ready metric tables (Markdown).

Input rows (header lines are written by the Makefile):
  cpu: impl,tier,batch,threads,tenants,trial,throughput,mean_us,p50_us,p99_us,verify_fail
  gpu: impl,kernel,gpu,tier,batch,tenants,trial,kernel_ms,throughput,mean_us,p50_us,p99_us,
       mean_attempts,warp_eff,failures,layout
"""

# ./benchmarks/score_results.py

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

# ML-DSA-44 expected signing attempts: 1 / P_accept, with P_accept ~= 0.2353 (FIPS 204 Table 1).
EXPECTED_ATTEMPTS_MLDSA44 = 4.25


def read_rows(path, first_col):
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for rec in csv.reader(f):
            if not rec or rec[0] == "impl" or rec[0] != first_col:
                continue
            rows.append(rec)
    return rows


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def fmt(m, s, digits=0):
    if math.isnan(m):
        return "n/a"
    return f"{m:,.{digits}f} ± {s:,.{digits}f}"


def gpu_table(gpu_rows, cpu_rows):
    # Group GPU rows by (gpu name, kernel, layout, batch).
    groups = defaultdict(list)
    for r in gpu_rows:
        gpu_name, kernel, layout, batch = r[2], r[1], r[15], int(r[4])
        groups[(gpu_name, kernel, layout, batch)].append(
            {
                "thr": float(r[8]),
                "mean": float(r[9]),
                "p50": float(r[10]),
                "p99": float(r[11]),
                "att": float(r[12]),
                "weff": float(r[13]),
                "fail": int(r[14]),
            }
        )

    cpu_groups = defaultdict(list)
    for r in cpu_rows:
        cpu_groups[int(r[2])].append(
            {"thr": float(r[6]), "mean": float(r[7]), "p50": float(r[8]), "p99": float(r[9]), "fail": int(r[10])}
        )

    lines = ["## GPU kernel results (ML-DSA-44)", ""]
    lines.append(
        "| GPU | Kernel | Layout | Batch | Throughput (sig/s) | p50 (us) | p99 (us) | "
        "Mean attempts | Warp efficiency | Failures |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for key in sorted(groups, key=lambda k: (k[0], k[3], k[1], k[2])):
        rs = groups[key]
        thr = mean_std([r["thr"] for r in rs])
        p50 = mean_std([r["p50"] for r in rs])
        p99 = mean_std([r["p99"] for r in rs])
        att = mean_std([r["att"] for r in rs])
        weff = mean_std([r["weff"] for r in rs])
        fail = sum(r["fail"] for r in rs)
        summary[key] = {"thr": thr[0], "p99": p99[0], "weff": weff[0], "att": att[0]}
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:,} | {fmt(*thr)} | {fmt(*p50)} | {fmt(*p99)} | "
            f"{att[0]:.3f} | {weff[0]:.3f} | {fail} |"
        )
    lines.append("")

    lines.append("## Comparisons")
    lines.append("")
    lines.append("| GPU | Batch | WarpSign vs naive throughput | p99 latency change | Warp eff. (naive -> WarpSign) | "
                 "WarpSign vs CPU throughput |")
    lines.append("|---|---:|---:|---:|---|---:|")
    gpus = sorted({k[0] for k in summary})
    batches = sorted({k[3] for k in summary})
    for g in gpus:
        for b in batches:
            nv = summary.get((g, "naive", "aos", b))
            ws = summary.get((g, "warpsign", "soa", b))
            if not nv or not ws:
                continue
            speed = ws["thr"] / nv["thr"]
            p99_change = (ws["p99"] - nv["p99"]) / nv["p99"] * 100.0
            cpu = cpu_groups.get(b)
            cpu_speed = f"{ws['thr'] / statistics.fmean([c['thr'] for c in cpu]):.2f}x" if cpu else "n/a"
            lines.append(
                f"| {g} | {b:,} | {speed:.2f}x | {p99_change:+.1f}% | "
                f"{nv['weff']:.3f} -> {ws['weff']:.3f} | {cpu_speed} |"
            )
    lines.append("")

    lines.append("## Validation")
    lines.append("")
    for key in sorted(summary, key=lambda k: (k[0], k[3], k[1], k[2])):
        att = summary[key]["att"]
        dev = (att - EXPECTED_ATTEMPTS_MLDSA44) / EXPECTED_ATTEMPTS_MLDSA44 * 100.0
        lines.append(
            f"- {key[0]} {key[1]} batch={key[3]:,}: mean attempts {att:.3f} vs expected "
            f"{EXPECTED_ATTEMPTS_MLDSA44} ({dev:+.1f}%)"
        )
    lines.append("")
    return lines


def cpu_table(cpu_rows):
    groups = defaultdict(list)
    for r in cpu_rows:
        groups[(r[1], int(r[2]), int(r[3]))].append(
            {"thr": float(r[6]), "mean": float(r[7]), "p50": float(r[8]), "p99": float(r[9]), "fail": int(r[10])}
        )
    lines = ["## CPU baseline (liboqs, OpenMP)", ""]
    lines.append("| Tier | Batch | Threads | Throughput (sig/s) | p50 (us) | p99 (us) | Verify failures |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        rs = groups[key]
        thr = mean_std([r["thr"] for r in rs])
        p50 = mean_std([r["p50"] for r in rs])
        p99 = mean_std([r["p99"] for r in rs])
        fail = max(r["fail"] for r in rs)
        lines.append(f"| {key[0]} | {key[1]:,} | {key[2]} | {fmt(*thr)} | {fmt(*p50, 1)} | {fmt(*p99, 1)} | {fail} |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default="results/cpu.csv")
    ap.add_argument("--gpu", default="results/gpu.csv")
    ap.add_argument("--out", default="results/metrics.md")
    args = ap.parse_args()

    cpu_rows = read_rows(args.cpu, "cpu")
    gpu_rows = read_rows(args.gpu, "gpu")

    out = ["# WarpSign-PQC metrics", "", "Mean ± std over recorded trials (warmup excluded).", ""]
    if cpu_rows:
        out += cpu_table(cpu_rows)
    if gpu_rows:
        out += gpu_table(gpu_rows, cpu_rows)
    if not cpu_rows and not gpu_rows:
        out.append("No result rows found.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
