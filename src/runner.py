"""
Multiprocess parallel simulation runner and benchmark reporter.
"""

import os
import glob
import time
import concurrent.futures

from .constants import (
    SCALE_FACTORS, MAX_DURATION, FIXED_FILE_SIZE_MB,
    BASELINE_METHODS, METHOD_LABELS,
)
from .orbit import precompute_constellation
from .traffic import generate_arrivals
from .sim_engine import run_sim


def purge_numba_cache():
    """
    Remove cached Numba .nbc/.nbi binaries before spawning worker processes
    to avoid cross-function inline cache staleness after source updates.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
    if os.path.isdir(cache_dir):
        for pattern in ("*.nbc", "*.nbi"):
            for path in glob.glob(os.path.join(cache_dir, pattern)):
                try:
                    os.remove(path)
                except OSError:
                    pass


def worker(task):
    """Execute a single (scale, method_id) simulation task."""
    scale, m, max_duration, raan0, phase0, vis_idx, vis_cnt, arr_t, arr_pos, arr_tgt = task
    t0 = time.time()
    result = run_sim(m, max_duration, raan0, phase0, vis_idx, vis_cnt, arr_t, arr_pos, arr_tgt)
    return scale, m, result, time.time() - t0


def run_experiments(
    scale_factors=SCALE_FACTORS,
    max_duration=MAX_DURATION,
    methods=BASELINE_METHODS,
    out_dir="experiments/latest_run",
):
    """Run baseline benchmark sweeps across traffic scales in parallel."""
    raan0, phase0, vis_idx, vis_cnt = precompute_constellation(max_duration)
    print("\n" + "=" * 116)
    print("LEO Satellite Event-Driven Network Simulator (Baseline Benchmark)")
    print("=" * 116)

    tasks = []
    scale_info = {}

    for scale in scale_factors:
        arr_t, arr_pos, arr_tgt = generate_arrivals(scale, max_duration, seed=42 + int(scale * 10))
        offered_gbps = len(arr_t) * FIXED_FILE_SIZE_MB * 8.0 / max_duration / 1000.0
        scale_info[scale] = {
            "arrivals": len(arr_t),
            "target_ues": int(arr_tgt.sum()),
            "offered_load": offered_gbps,
        }
        for m in methods:
            tasks.append((scale, m, max_duration, raan0, phase0, vis_idx, vis_cnt, arr_t, arr_pos, arr_tgt))

    print(f"Dispatching {len(tasks)} simulation tasks across CPU workers...")
    t_start_all = time.time()

    results = {scale: {} for scale in scale_factors}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            scale, m, r, elapsed = future.result()
            results[scale][m] = r
            print(f"  [OK] Scale x{scale} | {METHOD_LABELS[m]:<12} ({elapsed:.2f}s)")

    lines = [
        "=" * 116,
        "Simulation Benchmark Summary",
        "=" * 116,
    ]
    csv_rows = ["scale,method_id,method_name,avg_mbps,edge10_mbps,ho,hof,rlf,drop,block,unfin,heap_pk,evals,max_ev_ue"]

    for scale in scale_factors:
        info = scale_info[scale]
        lines.append(
            f"\nLoad Scale x{scale}: {info['arrivals']} arrivals, target UEs = {info['target_ues']}, "
            f"offered load ~ {info['offered_load']:.1f} Gbps, T = {max_duration:.0f}s"
        )
        lines.append(
            f"{'Method':<14} | {'Avg(Mbps)':>9} | {'Edge10%':>8} | {'HO':>6} | {'HOF':>5} | {'RLF':>5} | "
            f"{'Drop':>4} | {'Block':>5} | {'Unfin':>5} | {'HeapPk':>7} | {'Evals':>8} | {'MaxEv/UE':>8}"
        )
        lines.append("-" * 116)
        for m in methods:
            avg, edge, ho, hof, rlf, drop, block, cens, hpk, nev, mxev = results[scale][m]
            lines.append(
                f"{METHOD_LABELS[m]:<14} | {avg:>9.2f} | {edge:>8.2f} | {ho:>6} | {hof:>5} | {rlf:>5} | "
                f"{drop:>4} | {block:>5} | {cens:>5} | {hpk:>7} | {nev:>8} | {mxev:>8}"
            )
            csv_rows.append(
                f"{scale},{m},{METHOD_LABELS[m]},{avg:.2f},{edge:.2f},{ho},{hof},{rlf},{drop},{block},{cens},{hpk},{nev},{mxev}"
            )

    total_sec = time.time() - t_start_all
    lines.append(f"\nTotal wall-clock time: {total_sec:.2f}s")
    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(csv_rows) + "\n")
        print(f"Saved results to {out_dir}/results.csv and summary.txt")

    return results


def main():
    """CLI entry point."""
    import argparse

    purge_numba_cache()
    parser = argparse.ArgumentParser(
        description="Numba-accelerated Event-Driven LEO Network Simulator (Baseline Benchmark)"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=MAX_DURATION,
        help=f"Simulation duration [s] (default: {MAX_DURATION})",
    )
    parser.add_argument(
        "--scales", "-s", type=float, nargs="+", default=SCALE_FACTORS,
        help=f"Traffic scale factors (default: {SCALE_FACTORS})",
    )
    parser.add_argument(
        "--methods", "-m", type=int, nargs="+", default=list(BASELINE_METHODS),
        help="Baseline policy IDs (0:BHO, 1:Normal CHO, 2:Load-Min, 3:Load-Std)",
    )
    parser.add_argument(
        "--out-dir", "-o", type=str, default="experiments/latest_run",
        help="Output directory for results.csv and summary.txt",
    )
    args = parser.parse_args()

    run_experiments(
        scale_factors=args.scales,
        max_duration=args.duration,
        methods=args.methods,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
