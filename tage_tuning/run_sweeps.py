#!/usr/bin/env python3
"""Run the sensitivity sweeps and genetic-algorithm DSE on TAGE.

Uses the existing dse.py primitives (compile_one, run_one, evaluate_many).
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dse import DEFAULT, PARAM_NAMES, cfg_name, evaluate_many, load_cache, valid  # noqa

TUNE = Path(__file__).resolve().parent


def default_with(**overrides):
    p = list(DEFAULT)
    for k, v in overrides.items():
        p[PARAM_NAMES.index(k)] = v
    return tuple(p)


def sensitivity():
    """Vary each parameter independently around the default."""
    sweeps = {
        "NUMG":   [4, 6, 8, 10, 12, 14],
        "LOGG":   [9, 10, 11, 12, 13, 14],
        "LOGB":   [9, 10, 11, 12, 13, 14],
        "TAGW":   [7, 9, 11, 13, 15],
        "GHIST":  [20, 50, 100, 150, 200, 300],
        "LOGP1":  [10, 12, 14, 16],
        "GHIST1": [4, 6, 8, 12],
    }
    all_cfgs = set()
    by_sweep = {}
    for k, vs in sweeps.items():
        by_sweep[k] = []
        for v in vs:
            c = default_with(**{k: v})
            if valid(c):
                all_cfgs.add(c)
                by_sweep[k].append(c)
    all_cfgs = list(all_cfgs)
    print(f"Sensitivity: {len(all_cfgs)} unique configs")
    t0 = time.time()
    res = evaluate_many(all_cfgs, jobs=8)
    print(f"Done in {time.time()-t0:.1f}s")
    # Also include the default's results
    default_res = evaluate_many([DEFAULT], jobs=1)
    res.update(default_res)
    summary = {"default": {"params": dict(zip(PARAM_NAMES, DEFAULT)),
                           **res[DEFAULT]}}
    for k, cfgs in by_sweep.items():
        summary[k] = []
        for c in cfgs:
            if c not in res:
                continue
            summary[k].append({
                "params": dict(zip(PARAM_NAMES, c)),
                **res[c],
            })
    (TUNE / "sensitivity.json").write_text(json.dumps(summary, indent=2))
    # Print a quick summary
    print(f"\nDefault VFS={res[DEFAULT]['vfs']:.4f}  MPKI={res[DEFAULT]['mpki']:.3f}  E={res[DEFAULT]['epi']:.0f}")
    for k, cfgs in by_sweep.items():
        print(f"\n[{k} sweep]")
        for c in cfgs:
            if c not in res:
                continue
            m = res[c]
            v = c[PARAM_NAMES.index(k)]
            print(f"  {k}={v:4d}  MPKI={m['mpki']:.3f}  E={m['epi']:6.0f}  "
                  f"P2lat={m['p2_lat']}  VFS={m['vfs']:.4f}")


def ga(gens=6, pop=12, seed=42):
    random.seed(seed)
    ranges = {
        "LOGLB": [6],  # fixed
        "NUMG":  list(range(4, 15)),
        "LOGG":  list(range(9, 15)),
        "LOGB":  list(range(9, 15)),
        "TAGW":  list(range(7, 16)),
        "GHIST": [20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400],
        "LOGP1": list(range(10, 17)),
        "GHIST1": [4, 5, 6, 7, 8, 10, 12, 16],
    }

    def rand_indiv():
        d = {k: random.choice(v) for k, v in ranges.items()}
        return tuple(d[k] for k in PARAM_NAMES)

    def mutate(c):
        d = dict(zip(PARAM_NAMES, c))
        keys = random.sample(list(ranges.keys()), k=random.choice([1, 2]))
        for k in keys:
            if len(ranges[k]) == 1:
                continue
            d[k] = random.choice(ranges[k])
        return tuple(d[k] for k in PARAM_NAMES)

    def crossover(a, b):
        d = {}
        for k in PARAM_NAMES:
            ia = PARAM_NAMES.index(k)
            d[k] = random.choice([a[ia], b[ia]])
        return tuple(d[k] for k in PARAM_NAMES)

    # Seeds: default + a few educated guesses + random fill
    pop_list = [DEFAULT]
    pop_list.append(default_with(NUMG=12, LOGG=12, GHIST=200))
    pop_list.append(default_with(NUMG=10, LOGG=12, LOGB=13, TAGW=12, GHIST=150))
    pop_list.append(default_with(NUMG=6, LOGG=10, LOGB=10, GHIST=80))
    pop_list.append(default_with(NUMG=8, LOGG=13, LOGB=13, TAGW=12, GHIST=200))
    while len(pop_list) < pop:
        c = rand_indiv()
        if valid(c):
            pop_list.append(c)
    pop_list = [c for c in pop_list if valid(c)]

    history = []

    def score_pop(configs):
        uniq = list(set(configs))
        res = evaluate_many(uniq, jobs=8)
        return [(c, res[c]) for c in configs if c in res]

    scored = score_pop(pop_list)
    scored.sort(key=lambda x: x[1]["vfs"], reverse=True)
    history.append({
        "gen": 0,
        "best_vfs": scored[0][1]["vfs"],
        "best_cfg": dict(zip(PARAM_NAMES, scored[0][0])),
        "median_vfs": scored[len(scored) // 2][1]["vfs"],
        "all": [{"cfg": dict(zip(PARAM_NAMES, c)), **m} for c, m in scored],
    })
    print(f"\nGen 0  best VFS={scored[0][1]['vfs']:.4f}  MPKI={scored[0][1]['mpki']:.3f}  "
          f"median={scored[len(scored)//2][1]['vfs']:.4f}")

    for g in range(1, gens + 1):
        elite_n = max(2, pop // 3)
        elites = [c for c, _ in scored[:elite_n]]
        next_pop = list(elites)
        tries = 0
        while len(next_pop) < pop and tries < pop * 10:
            tries += 1
            a, b = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child = crossover(a, b)
            if random.random() < 0.7:
                child = mutate(child)
            if not valid(child):
                continue
            next_pop.append(child)
        # Fill with random
        while len(next_pop) < pop:
            c = rand_indiv()
            if valid(c):
                next_pop.append(c)
        scored = score_pop(next_pop)
        scored.sort(key=lambda x: x[1]["vfs"], reverse=True)
        history.append({
            "gen": g,
            "best_vfs": scored[0][1]["vfs"],
            "best_cfg": dict(zip(PARAM_NAMES, scored[0][0])),
            "median_vfs": scored[len(scored) // 2][1]["vfs"],
            "all": [{"cfg": dict(zip(PARAM_NAMES, c)), **m} for c, m in scored],
        })
        print(f"Gen {g}  best VFS={scored[0][1]['vfs']:.4f}  MPKI={scored[0][1]['mpki']:.3f}  "
              f"median={scored[len(scored)//2][1]['vfs']:.4f}  "
              f"cfg={history[-1]['best_cfg']}")

    (TUNE / "ga_history.json").write_text(json.dumps(history, indent=2))
    print("\nGA done. Best overall:")
    print(json.dumps(history[-1]["best_cfg"], indent=2))
    m = history[-1]["all"][0]
    print(f"VFS={m['vfs']:.4f}  MPKI={m['mpki']:.3f}  EPI={m['epi']:.0f}  "
          f"P1lat={m['p1_lat']}  P2lat={m['p2_lat']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sensitivity", "ga", "both"])
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.cmd in ("sensitivity", "both"):
        sensitivity()
    if args.cmd in ("ga", "both"):
        ga(gens=args.gens, pop=args.pop, seed=args.seed)
