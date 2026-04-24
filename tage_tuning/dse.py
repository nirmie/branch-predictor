#!/usr/bin/env python3
"""TAGE DSE driver. Compile/run/score TAGE configs against gcc_test_trace.

Each config is a tuple: (LOGLB, NUMG, LOGG, LOGB, TAGW, GHIST, LOGP1, GHIST1).
Defaults match tage<>: (6,8,11,12,11,100,14,6).

Caches results in results.csv keyed by config so the same config is never
re-evaluated. evaluate_many() compiles+runs in parallel.
"""

import csv
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNE = Path(__file__).resolve().parent
BINS = TUNE / "bins"
RES = TUNE / "results"
CACHE = TUNE / "results.csv"
TRACE = ROOT / "gcc_test_trace.gz"
WARMUP = 1_000_000
MEASURE = 40_000_000

PARAM_NAMES = ("LOGLB", "NUMG", "LOGG", "LOGB", "TAGW", "GHIST", "LOGP1", "GHIST1")
DEFAULT = (6, 8, 11, 12, 11, 100, 14, 6)


def cfg_str(c):
    return ",".join(str(x) for x in c)


def cfg_name(c):
    return "tage_" + "_".join(str(x) for x in c)


def valid(c):
    LOGLB, NUMG, LOGG, LOGB, TAGW, GHIST, LOGP1, GHIST1 = c
    if LOGLB <= 2 or NUMG <= 0:
        return False
    LOGLINEINST = LOGLB - 2
    if LOGP1 <= LOGLINEINST:
        return False
    if LOGB <= LOGLINEINST:
        return False
    if TAGW <= LOGLINEINST:
        return False
    if GHIST <= 1 or GHIST1 < 1:
        return False
    return True


def compile_one(c):
    """Compile config c with ./compile. Returns path to binary or None."""
    name = cfg_name(c)
    bin_path = BINS / name
    if bin_path.exists():
        return str(bin_path)
    tpl = f"tage<{cfg_str(c)}>"
    cmd = ["./compile", "cbp", f"-DPREDICTOR={tpl}", "-o", str(bin_path)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not bin_path.exists():
        return None
    return str(bin_path)


def run_one(c, bin_path):
    """Run binary and compute VFS. Returns dict of metrics or None."""
    name = cfg_name(c)
    out_dir = RES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gcc_test.out"
    cmd = [bin_path, str(TRACE), "test", str(WARMUP), str(MEASURE)]
    with open(out_file, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        return None
    line = out_file.read_text().strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) != 12:
        return None
    instructions = float(parts[1])
    pred_cycles = float(parts[4])
    extra_cycles = float(parts[5])
    divergences = float(parts[6])
    div_at_end = float(parts[7])
    misp = float(parts[8])
    p1_lat = math.ceil(float(parts[9]))
    p2_lat = math.ceil(float(parts[10]))
    epi = float(parts[11])

    if p2_lat <= p1_lat:
        cycles = pred_cycles * max(1, p2_lat)
    else:
        cycles = pred_cycles * max(1, p1_lat) + divergences * p2_lat - div_at_end * max(1, p1_lat)
    cycles += extra_cycles
    ipc = instructions / cycles
    mpi = misp / instructions
    p2_to_exec = 9
    cpi = mpi * (p2_to_exec + p2_lat - max(1, min(p1_lat, p2_lat)))
    mpki = mpi * 1000.0

    vfs = compute_vfs(ipc, cpi, epi)
    return {
        "ipc": ipc, "cpi": cpi, "epi": epi, "mpki": mpki,
        "p1_lat": p1_lat, "p2_lat": p2_lat,
        "vfs": vfs,
    }


def compute_vfs(IPCcbp, CPIcbp, EPIcbp):
    IPCcbp0 = 8
    CPIcbp0 = 0.0315
    EPIcbp0 = 1000
    ALPHA = 1.625
    BETA = 4 * ALPHA / (ALPHA - 1) ** 2
    cbp_energy_ratio = 0.05
    EPI0 = EPIcbp0 / cbp_energy_ratio
    GAMMA = 2 / (ALPHA - 1)

    WPI0 = IPCcbp0 * CPIcbp0
    WPI = IPCcbp * CPIcbp
    speedup = (IPCcbp / IPCcbp0) * (1 + WPI0) / (1 + WPI)
    LAMBDA = 1 / (1 + WPI0 / 2) - cbp_energy_ratio
    normEPI = ((EPIcbp / EPIcbp0) * cbp_energy_ratio + LAMBDA * speedup ** GAMMA) * (1 + WPI / 2)
    return speedup * ALPHA * (1 - 2 / (1 + math.sqrt(1 + BETA / (speedup * normEPI))))


def load_cache():
    if not CACHE.exists():
        return {}
    out = {}
    with open(CACHE) as f:
        r = csv.DictReader(f)
        for row in r:
            key = tuple(int(row[n]) for n in PARAM_NAMES)
            out[key] = {
                "vfs": float(row["vfs"]),
                "mpki": float(row["mpki"]),
                "epi": float(row["epi"]),
                "ipc": float(row["ipc"]),
                "cpi": float(row["cpi"]),
                "p1_lat": int(row["p1_lat"]),
                "p2_lat": int(row["p2_lat"]),
            }
    return out


def append_cache(c, m):
    new_file = not CACHE.exists()
    with open(CACHE, "a") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(list(PARAM_NAMES) + ["vfs", "mpki", "epi", "ipc", "cpi", "p1_lat", "p2_lat"])
        w.writerow(list(c) + [
            f"{m['vfs']:.6f}", f"{m['mpki']:.4f}", f"{m['epi']:.2f}",
            f"{m['ipc']:.6f}", f"{m['cpi']:.6f}", m['p1_lat'], m['p2_lat']
        ])


def _eval_worker(c):
    bin_path = compile_one(c)
    if bin_path is None:
        return c, None
    m = run_one(c, bin_path)
    return c, m


def evaluate_many(configs, jobs=8):
    cache = load_cache()
    new_results = {}
    todo = []
    for c in configs:
        if not valid(c):
            continue
        if c in cache:
            new_results[c] = cache[c]
            continue
        if c in new_results:
            continue
        todo.append(c)
    if todo:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_eval_worker, c): c for c in todo}
            for fut in as_completed(futs):
                c, m = fut.result()
                if m is not None:
                    append_cache(c, m)
                    new_results[c] = m
                else:
                    print(f"FAILED: {c}", file=sys.stderr)
    return new_results


def evaluate(c):
    return evaluate_many([c]).get(c)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="comma-separated 8 ints")
    args = ap.parse_args()
    if args.config:
        c = tuple(int(x) for x in args.config.split(","))
    else:
        c = DEFAULT
    print(f"Evaluating tage<{cfg_str(c)}> ...")
    m = evaluate(c)
    print(m)
