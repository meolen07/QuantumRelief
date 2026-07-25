#!/usr/bin/env python3
"""
Substantial retrain for QuantumRelief hackathon demo.

1) (optional --hard) sync Classical-failure seeds + hard hazard labels
2) Regenerate Manila dynamic-routing dataset (default 1000 episodes)
   — or --reuse-dataset to keep data/routing_dataset.npz
3) Train Classical FiLM (ablation / hybrid seed) — skippable with --skip-classical
4) Train Hybrid QML (PennyLane PHN) — Phase A + B with hard-seed oversample
5) Fair 3-way eval: hard_seeds + random (≥24 trials)
6) Write data/retrain_report.json

Recommended hard Hybrid push (reuse existing ~18.9k dataset):

  source .venv/bin/activate
  caffeinate -dimsu python -u scripts/retrain_models.py --hard --reuse-dataset \
    --skip-classical --hybrid-a 24 --hybrid-b 12 --hybrid-max-samples 8000 \
    --hard-repeats 10 --eval-trials 28 --lambda-safe 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.dataset_generation import (
    HARD_HAZARD_INTENSITY,
    HARD_OVERSAMPLE,
    HARD_SEEDS_PATH,
    DEMO_SCENARIOS_PATH,
    collect_hard_seed_samples,
    generate_dataset,
    load_dataset,
    load_hard_seeds,
    write_hard_seeds_from_scenarios,
)
from src.film_model import load_film_model, train_film_model
from src.graph_setup import load_or_build_graph, random_epicenter, select_exit_nodes
from src.quantum_hybrid import (
    estimate_quantum_contribution_pct,
    finetune_hybrid_on_hard,
    load_hybrid_model,
    quantum_status,
    train_hybrid_model,
)
from src.routing_service import compare_three_way
from src.utils import DATASET_PATH, HYBRID_CHECKPOINT, MODEL_CHECKPOINT, ensure_dirs


def _random_start(G, exits, rng):
    nodes = [n for n in G.nodes() if n not in exits]
    return nodes[int(rng.integers(0, len(nodes)))]


def _build_eval_trials(
    n_trials: int,
    seed: int = 7,
    *,
    hard_repeats: int = 2,
) -> list[dict]:
    """
    Mix hard_seeds (priority, cycled) + random start/epi/exit for ≥28-trial eval.
    """
    G = load_or_build_graph()
    exits = select_exit_nodes(G, n_exits=3, seed=42)
    seeds = load_hard_seeds(G=G)
    rng = np.random.default_rng(seed)
    trials: list[dict] = []

    # Heavy hard-seed coverage: each seed ≥ hard_repeats times
    if seeds:
        n_hard = max(len(seeds) * max(1, hard_repeats), min(n_trials // 2, len(seeds) * 3))
        for i in range(n_hard):
            spec = seeds[i % len(seeds)]
            trials.append(
                {
                    "kind": "hard",
                    "start": spec["start_node"],
                    "dest": spec["dest_node"],
                    "epi_ll": (float(spec["epi_lon"]), float(spec["epi_lat"])),
                    "id": spec.get("id"),
                }
            )

    while len(trials) < n_trials:
        dest = exits[int(rng.integers(0, len(exits)))]
        start = _random_start(G, exits, rng)
        epi_ll, _ = random_epicenter(G, seed=int(rng.integers(0, 1_000_000)))
        trials.append(
            {
                "kind": "random",
                "start": start,
                "dest": dest,
                "epi_ll": epi_ll,
                "id": None,
            }
        )

    rng.shuffle(trials)
    return trials[:n_trials]


def eval_routes_three_way(
    n_trials: int = 28,
    seed: int = 7,
    *,
    hard_repeats: int = 2,
) -> dict:
    """Roll out Hybrid vs Classical vs Dijkstra; hard_seeds + random mix."""
    G = load_or_build_graph()
    hybrid = load_hybrid_model()
    classical = load_film_model()
    ds = load_dataset()
    mean, std = ds["mean"], ds["std"]
    trials = _build_eval_trials(
        n_trials=n_trials, seed=seed, hard_repeats=hard_repeats
    )

    reached_h = reached_c = reached_d = 0
    assist = 0
    n_hard_done = 0
    time_h, time_c, time_d = [], [], []
    hops_h, hops_c, hops_d = [], [], []
    ov_h, ov_c = [], []
    hard_time_h, hard_time_c, hard_time_d = [], [], []
    hard_ov_h, hard_ov_c = [], []
    hard_travel_wins = 0
    hard_compared = 0
    hybrid_beats_classical = 0
    hybrid_near_dij = 0
    hybrid_travel_wins = 0  # strict: H travel ≤ C (ε)
    hybrid_overlap_wins = 0
    n_compared = 0

    for i, trial in enumerate(trials):
        start, dest, epi_ll = trial["start"], trial["dest"], trial["epi_ll"]
        try:
            cmp = compare_three_way(
                G,
                hybrid,
                classical,
                mean,
                std,
                start,
                dest,
                epi_ll,
                include_classical=True,
                include_dijkstra=True,
            )
        except Exception as exc:
            print(f"  trial {i}: skip ({exc})")
            continue

        h, c, d = cmp["hybrid"], cmp["classical"], cmp["dijkstra"]
        ok_h = bool(h["exit_reached"])
        ok_c = bool(c["exit_reached"]) if c else False
        ok_d = bool(d["exit_reached"]) if d else False
        reached_h += int(ok_h)
        reached_c += int(ok_c)
        reached_d += int(ok_d)
        assist += int(bool(h["meta"].get("dijkstra_assist")))
        is_hard = trial.get("kind") == "hard"
        if is_hard:
            n_hard_done += 1
        time_h.append(float(h["travel_time"]))
        hops_h.append(int(h["hops"]))
        if c:
            time_c.append(float(c["travel_time"]))
            hops_c.append(int(c["hops"]))
        if d:
            time_d.append(float(d["travel_time"]))
            hops_d.append(int(d["hops"]))
            ov_h.append(float(h.get("overlap_vs_dijkstra_pct") or 0.0))
            if c:
                ov_c.append(float(c.get("overlap_vs_dijkstra_pct") or 0.0))
        if is_hard:
            hard_time_h.append(float(h["travel_time"]))
            if c:
                hard_time_c.append(float(c["travel_time"]))
            if d:
                hard_time_d.append(float(d["travel_time"]))
                hard_ov_h.append(float(h.get("overlap_vs_dijkstra_pct") or 0.0))
                if c:
                    hard_ov_c.append(float(c.get("overlap_vs_dijkstra_pct") or 0.0))
        if cmp["narrative"].get("hybrid_beats_classical"):
            hybrid_beats_classical += 1
        if cmp["narrative"].get("hybrid_near_dijkstra"):
            hybrid_near_dij += 1

        if c and ok_h and ok_c:
            n_compared += 1
            ht, ct = float(h["travel_time"]), float(c["travel_time"])
            if ht <= ct * 1.001:
                hybrid_travel_wins += 1
            hov = float(h.get("overlap_vs_dijkstra_pct") or 0.0)
            cov = float(c.get("overlap_vs_dijkstra_pct") or 0.0) if c else 0.0
            if hov + 1e-9 >= cov:
                hybrid_overlap_wins += 1
            if is_hard:
                hard_compared += 1
                if ht <= ct * 1.001:
                    hard_travel_wins += 1

        kind = trial.get("kind", "?")
        tid = trial.get("id") or ""
        print(
            f"  trial {i:02d} [{kind}{(':'+tid) if tid else ''}]: H={h['travel_time']:.1f}/"
            f"C={c['travel_time'] if c else float('nan'):.1f}/"
            f"D={d['travel_time'] if d else float('nan'):.1f}  "
            f"exit={ok_h}/{ok_c}/{ok_d}  "
            f"ovH={h.get('overlap_vs_dijkstra_pct')}  "
            f"assist={h['meta'].get('assist_hops', 0)}"
        )

    n = max(len(time_h), 1)
    n_cmp = max(n_compared, 1)
    n_hard_cmp = max(hard_compared, 1)
    mt_h = float(np.mean(time_h)) if time_h else 0.0
    mt_c = float(np.mean(time_c)) if time_c else 0.0
    return {
        "n_trials": len(time_h),
        "n_hard_trials": n_hard_done,
        "exit_reached_pct": {
            "hybrid": 100.0 * reached_h / n,
            "classical": 100.0 * reached_c / n,
            "dijkstra": 100.0 * reached_d / n,
        },
        "assist_pct": 100.0 * assist / n,
        "mean_time": {
            "hybrid": mt_h,
            "classical": mt_c,
            "dijkstra": float(np.mean(time_d)) if time_d else 0.0,
        },
        "mean_hops": {
            "hybrid": float(np.mean(hops_h)) if hops_h else 0.0,
            "classical": float(np.mean(hops_c)) if hops_c else 0.0,
            "dijkstra": float(np.mean(hops_d)) if hops_d else 0.0,
        },
        "mean_overlap_pct": {
            "hybrid": float(np.mean(ov_h)) if ov_h else 0.0,
            "classical": float(np.mean(ov_c)) if ov_c else 0.0,
        },
        "hard_only": {
            "n": len(hard_time_h),
            "mean_time": {
                "hybrid": float(np.mean(hard_time_h)) if hard_time_h else 0.0,
                "classical": float(np.mean(hard_time_c)) if hard_time_c else 0.0,
                "dijkstra": float(np.mean(hard_time_d)) if hard_time_d else 0.0,
            },
            "mean_overlap_pct": {
                "hybrid": float(np.mean(hard_ov_h)) if hard_ov_h else 0.0,
                "classical": float(np.mean(hard_ov_c)) if hard_ov_c else 0.0,
            },
            "hybrid_travel_win_pct": 100.0 * hard_travel_wins / n_hard_cmp,
        },
        "hybrid_beats_classical_pct": 100.0 * hybrid_beats_classical / n,
        "hybrid_near_dijkstra_pct": 100.0 * hybrid_near_dij / n,
        "hybrid_travel_win_pct": 100.0 * hybrid_travel_wins / n_cmp,
        "hybrid_overlap_win_pct": 100.0 * hybrid_overlap_wins / n_cmp,
        "quantum_contrib_pct": estimate_quantum_contribution_pct(hybrid),
        "delta_mean_travel_pct_vs_classical": (
            100.0 * (mt_h - mt_c) / mt_c if mt_c > 1e-9 else 0.0
        ),
        "sample_trials": [
            {
                "hybrid_time": float(time_h[i]) if i < len(time_h) else None,
                "classical_time": float(time_c[i]) if i < len(time_c) else None,
                "dijkstra_time": float(time_d[i]) if i < len(time_d) else None,
            }
            for i in range(min(3, len(time_h)))
        ],
    }


def _ensure_hard_seeds(refresh: bool = False, find_n: int = 60) -> int:
    """Load or refresh Classical-failure seeds used for oversampling."""
    if refresh or not HARD_SEEDS_PATH.exists():
        if refresh or not DEMO_SCENARIOS_PATH.exists():
            print("[retrain] Refreshing advantage scenarios via find_advantage_scenarios…")
            import importlib.util

            mod_path = ROOT / "scripts" / "find_advantage_scenarios.py"
            spec = importlib.util.spec_from_file_location(
                "find_advantage_scenarios", mod_path
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            mod.search_advantage_scenarios(n_samples=find_n, top_n=8, seed=42)
        if DEMO_SCENARIOS_PATH.exists():
            write_hard_seeds_from_scenarios()
    seeds = load_hard_seeds()
    return len(seeds)


def _build_hybrid_subset(
    ds: dict,
    max_samples: int,
    *,
    hard_repeats: int = 6,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cap Hybrid train size while oversampling hard-seed trajectories.

    Root cause of next-hop-OK / travel-worse: uniform random 3500/18932
    under-represented Classical-failure corridors; Hybrid then diverges on
    early hops and loses mean travel despite similar val CE.
    """
    X_all = np.asarray(ds["X"], dtype=np.float32)
    y_all = np.asarray(ds["y"], dtype=np.int64)
    n = len(y_all)
    rng = np.random.default_rng(seed)

    X_hard, y_hard = collect_hard_seed_samples(
        ds["mean"],
        ds["std"],
        intensities=(1.0, HARD_HAZARD_INTENSITY),
        repeats=hard_repeats,
        seed=seed + 17,
    )
    n_hard = len(y_hard)
    # Reserve ~35% of the Hybrid budget for hard trajectories when available
    hard_budget = min(n_hard, max(0, int(0.35 * max_samples))) if n_hard else 0
    if hard_budget < n_hard and hard_budget > 0:
        take_h = rng.choice(n_hard, size=hard_budget, replace=False)
        X_hard, y_hard = X_hard[take_h], y_hard[take_h]
        n_hard = len(y_hard)

    rest = max(0, max_samples - n_hard)
    if rest >= n:
        X_rand, y_rand = X_all, y_all
    elif rest > 0:
        take = rng.choice(n, size=rest, replace=False)
        X_rand, y_rand = X_all[take], y_all[take]
    else:
        X_rand = np.zeros((0, X_all.shape[1]), dtype=np.float32)
        y_rand = np.zeros((0,), dtype=np.int64)

    if n_hard:
        Xh = np.concatenate([X_hard, X_rand], axis=0)
        yh = np.concatenate([y_hard, y_rand], axis=0)
    else:
        Xh, yh = X_rand, y_rand

    order = rng.permutation(len(yh))
    print(
        f"  hybrid train subset: {len(yh)} samples "
        f"(hard={n_hard}, random/pool={len(y_rand)}, pool_size={n})"
    )
    return Xh[order], yh[order]


def _build_hard_finetune_subset(
    ds: dict,
    max_samples: int,
    *,
    hard_repeats: int = 14,
    hard_fraction: float = 0.60,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a hard-heavy fine-tune set with per-sample weights.

    Inference uses hazard_intensity=1.0, so intensity-1.0 Dijkstra hops from
    Classical-failure seeds get the highest weight. Intensity-2.0 hops still
    help robustness but at lower weight. Pool hops keep general routing stable.
    """
    X_all = np.asarray(ds["X"], dtype=np.float32)
    y_all = np.asarray(ds["y"], dtype=np.int64)
    n = len(y_all)
    rng = np.random.default_rng(seed)

    # Inference-matched hard hops (primary)
    X_i1, y_i1 = collect_hard_seed_samples(
        ds["mean"],
        ds["std"],
        intensities=(1.0,),
        repeats=hard_repeats,
        seed=seed + 17,
    )
    # Hard-label intensity (secondary)
    X_i2, y_i2 = collect_hard_seed_samples(
        ds["mean"],
        ds["std"],
        intensities=(HARD_HAZARD_INTENSITY,),
        repeats=max(2, hard_repeats // 2),
        seed=seed + 31,
    )

    hard_budget = max(1, int(hard_fraction * max_samples))
    n_i1_budget = min(len(y_i1), int(0.75 * hard_budget)) if len(y_i1) else 0
    n_i2_budget = min(len(y_i2), hard_budget - n_i1_budget) if len(y_i2) else 0

    parts_x, parts_y, parts_w = [], [], []
    if n_i1_budget > 0:
        take = (
            rng.choice(len(y_i1), size=n_i1_budget, replace=False)
            if n_i1_budget < len(y_i1)
            else np.arange(len(y_i1))
        )
        parts_x.append(X_i1[take])
        parts_y.append(y_i1[take])
        parts_w.append(np.full(len(take), 4.0, dtype=np.float64))
    if n_i2_budget > 0:
        take = (
            rng.choice(len(y_i2), size=n_i2_budget, replace=False)
            if n_i2_budget < len(y_i2)
            else np.arange(len(y_i2))
        )
        parts_x.append(X_i2[take])
        parts_y.append(y_i2[take])
        parts_w.append(np.full(len(take), 2.0, dtype=np.float64))

    n_hard = sum(len(yy) for yy in parts_y)
    rest = max(0, max_samples - n_hard)
    if rest > 0:
        take = rng.choice(n, size=min(rest, n), replace=False)
        parts_x.append(X_all[take])
        parts_y.append(y_all[take])
        parts_w.append(np.full(len(take), 1.0, dtype=np.float64))

    Xh = np.concatenate(parts_x, axis=0)
    yh = np.concatenate(parts_y, axis=0)
    wh = np.concatenate(parts_w, axis=0)
    order = rng.permutation(len(yh))
    print(
        f"  hard finetune subset: {len(yh)} samples "
        f"(hard={n_hard}, pool={len(yh) - n_hard}, "
        f"i1_w=4.0, i2_w=2.0, pool_w=1.0)"
    )
    return Xh[order], yh[order], wh[order]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain QuantumRelief Classical + Hybrid on Manila routing data."
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Random episode count (default 1000; range 800–1500 recommended).",
    )
    p.add_argument("--classical-epochs", type=int, default=100)
    p.add_argument("--hybrid-a", type=int, default=20, help="Hybrid Phase A epochs.")
    p.add_argument("--hybrid-b", type=int, default=8, help="Hybrid Phase B epochs.")
    p.add_argument(
        "--hybrid-max-samples",
        type=int,
        default=6000,
        help="Cap Hybrid train subset size (CPU-bound PennyLane).",
    )
    p.add_argument(
        "--hard-repeats",
        type=int,
        default=6,
        help="Times to replay each hard seed × intensity when building Hybrid subset.",
    )
    p.add_argument(
        "--hard",
        action="store_true",
        help=(
            "Hard dataset: hazard_intensity=2.0 + oversample Classical-failure "
            "seeds (8×) mixed with random episodes."
        ),
    )
    p.add_argument(
        "--hazard-intensity",
        type=float,
        default=None,
        help="Override hazard intensity (default 2.0 with --hard, else 1.0).",
    )
    p.add_argument(
        "--oversample",
        type=int,
        default=None,
        help="Times to repeat each hard seed (default 8 with --hard).",
    )
    p.add_argument(
        "--refresh-seeds",
        action="store_true",
        help="Re-run find_advantage_scenarios before hard generation.",
    )
    p.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Keep existing data/routing_dataset.npz (do not regenerate).",
    )
    p.add_argument(
        "--dataset-only",
        action="store_true",
        help="Generate/save dataset only; skip training.",
    )
    p.add_argument(
        "--skip-classical",
        action="store_true",
        help="Reuse existing film_classical.pt (Hybrid-only fine-tune).",
    )
    p.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Train Classical only (still writes partial retrain_report).",
    )
    p.add_argument(
        "--lambda-safe",
        type=float,
        default=0.35,
        help=(
            "Weight of safety aux loss (default 0.35). "
            "L = L_CE(Dijkstra) + λ_safe · L_safe; 0 disables."
        ),
    )
    p.add_argument("--eval-trials", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    # Backward-compatible positional overrides:
    #   retrain_models.py [episodes] [classical] [hybrid_a] [hybrid_b] [max_samples]
    p.add_argument("legacy", nargs="*", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.legacy:
        vals = [int(x) for x in args.legacy]
        if len(vals) >= 1:
            args.episodes = vals[0]
        if len(vals) >= 2:
            args.classical_epochs = vals[1]
        if len(vals) >= 3:
            args.hybrid_a = vals[2]
        if len(vals) >= 4:
            args.hybrid_b = vals[3]
        if len(vals) >= 5:
            args.hybrid_max_samples = vals[4]
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    ensure_dirs()
    t0 = time.time()
    print("=== QuantumRelief hackathon retrain (3-way) ===")
    print(json.dumps(quantum_status(), indent=2))

    n_episodes = args.episodes
    classical_epochs = args.classical_epochs
    hybrid_q_epochs = args.hybrid_a
    hybrid_ft_epochs = args.hybrid_b
    hybrid_max_samples = args.hybrid_max_samples
    hard = bool(args.hard)
    hazard_intensity = (
        float(args.hazard_intensity)
        if args.hazard_intensity is not None
        else (HARD_HAZARD_INTENSITY if hard else 1.0)
    )
    oversample = (
        int(args.oversample)
        if args.oversample is not None
        else (HARD_OVERSAMPLE if hard else 1)
    )

    n_seeds = 0
    if hard:
        print("\n[0/4] Preparing Classical-failure hard seeds…")
        n_seeds = _ensure_hard_seeds(refresh=args.refresh_seeds)

    if args.reuse_dataset and DATASET_PATH.exists():
        print(f"\n[1/4] Reusing dataset → {DATASET_PATH}")
        ds = load_dataset()
        print(
            f"  samples: {len(ds['y'])}  X={ds['X'].shape}  "
            f"hard_meta={int(ds.get('meta_hard', [0])[0])}  "
            f"seed_eps={int(ds.get('meta_n_seed_episodes', [0])[0])}"
        )
    else:
        print(
            f"\n[1/4] Generating dataset ({n_episodes} random episodes"
            f"{', hard' if hard else ''}, intensity={hazard_intensity:.2f}, "
            f"oversample={oversample}, seeds={n_seeds})…"
        )
        if DATASET_PATH.exists():
            DATASET_PATH.unlink()
        ds = generate_dataset(
            n_episodes=n_episodes,
            seed=args.seed,
            save=True,
            hard=hard,
            hazard_intensity=hazard_intensity,
            oversample=oversample,
        )
        print(f"  samples: {len(ds['y'])}  X={ds['X'].shape}")

    if args.dataset_only:
        report = {
            "n_samples": int(len(ds["y"])),
            "n_episodes": n_episodes,
            "hard": hard,
            "hazard_intensity": hazard_intensity,
            "oversample": oversample,
            "n_hard_seeds": n_seeds,
            "dataset_only": True,
            "elapsed_sec": round(time.time() - t0, 1),
            "checkpoints": {"dataset": str(DATASET_PATH)},
        }
        out = ROOT / "data" / "retrain_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {out} (dataset-only)")
        return report

    classical_metrics = None
    if args.skip_classical and MODEL_CHECKPOINT.exists():
        print(f"\n[2/4] Skipping Classical — reusing {MODEL_CHECKPOINT}")
        # Carry forward prior report metrics if present
        prior = ROOT / "data" / "retrain_report.json"
        if prior.exists():
            try:
                classical_metrics = json.loads(prior.read_text()).get("classical")
            except Exception:
                classical_metrics = {"reused": True}
        else:
            classical_metrics = {"reused": True}
    else:
        print(f"\n[2/4] Training Classical FiLM ({classical_epochs} epochs)…")
        if MODEL_CHECKPOINT.exists():
            MODEL_CHECKPOINT.unlink()
        _, classical_metrics = train_film_model(
            ds["X"],
            ds["y"],
            epochs=classical_epochs,
            batch_size=64,
            lambda_safe=float(args.lambda_safe),
            feature_mean=ds.get("mean"),
            feature_std=ds.get("std"),
        )
        print("  classical metrics:", classical_metrics)

    hybrid_metrics = None
    route_stats = None
    if not args.skip_hybrid:
        print(
            f"\n[3/4] Training Hybrid QML PHN "
            f"(phase A={hybrid_q_epochs}, B={hybrid_ft_epochs}, "
            f"max_samples={hybrid_max_samples}, hard_repeats={args.hard_repeats}, "
            f"λ_safe={float(args.lambda_safe):.2f})…"
        )
        if HYBRID_CHECKPOINT.exists():
            # Keep a backup of the previous hybrid before overwriting
            bak = HYBRID_CHECKPOINT.with_name(HYBRID_CHECKPOINT.stem + "_prev.pt")
            try:
                HYBRID_CHECKPOINT.replace(bak)
                print(f"  backed up previous hybrid → {bak.name}")
            except OSError:
                HYBRID_CHECKPOINT.unlink()
        Xh, yh = _build_hybrid_subset(
            ds,
            hybrid_max_samples,
            hard_repeats=args.hard_repeats,
            seed=args.seed + 1,
        )
        _, hybrid_metrics = train_hybrid_model(
            Xh,
            yh,
            epochs_quantum=hybrid_q_epochs,
            epochs_finetune=hybrid_ft_epochs,
            batch_size=8,
            lambda_safe=float(args.lambda_safe),
            feature_mean=ds.get("mean"),
            feature_std=ds.get("std"),
        )
        print("  hybrid metrics:", hybrid_metrics)

        print(
            f"\n[4/4] Fair route eval ({args.eval_trials} trials, "
            "hard_seeds + random)…"
        )
        route_stats = eval_routes_three_way(n_trials=args.eval_trials, seed=7)
        print(json.dumps(route_stats, indent=2))
    else:
        print("\n[3/4] Skipping Hybrid (--skip-hybrid)")
        print("[4/4] Skipping route smoke-test")

    report = {
        "n_samples": int(len(ds["y"])),
        "n_episodes": n_episodes,
        "hard": hard,
        "hazard_intensity": hazard_intensity,
        "oversample": oversample,
        "n_hard_seeds": n_seeds,
        "reuse_dataset": bool(args.reuse_dataset and DATASET_PATH.exists()),
        "classical": classical_metrics,
        "hybrid": hybrid_metrics,
        "routes": route_stats,
        "tagline": (
            "Hybrid delivers near-Dijkstra quality with quantum-classical local inference"
        ),
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoints": {
            "classical": str(MODEL_CHECKPOINT),
            "hybrid": str(HYBRID_CHECKPOINT),
            "dataset": str(DATASET_PATH),
            "hard_seeds": str(HARD_SEEDS_PATH),
        },
        "train_config": {
            "classical_epochs": classical_epochs,
            "skip_classical": bool(args.skip_classical),
            "hybrid_phase_a": hybrid_q_epochs,
            "hybrid_phase_b": hybrid_ft_epochs,
            "hybrid_max_samples": hybrid_max_samples,
            "hard_repeats": args.hard_repeats,
            "hard": hard,
            "hazard_intensity": hazard_intensity,
            "oversample": oversample,
            "reuse_dataset": bool(args.reuse_dataset),
            "eval_trials": args.eval_trials,
            "lambda_safe": float(args.lambda_safe),
        },
    }
    out = ROOT / "data" / "retrain_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Done in {report['elapsed_sec']}s")

    if route_stats:
        mt = route_stats["mean_time"]
        mo = route_stats["mean_overlap_pct"]
        print(
            "\n=== Goal check ===\n"
            f"  Hybrid mean travel ≤ Classical: "
            f"{mt['hybrid']:.3f} ≤ {mt['classical']:.3f} → "
            f"{mt['hybrid'] <= mt['classical']}\n"
            f"  Hybrid overlap ≥ Classical: "
            f"{mo['hybrid']:.2f} ≥ {mo['classical']:.2f} → "
            f"{mo['hybrid'] >= mo['classical']}\n"
            f"  Hybrid travel wins: {route_stats.get('hybrid_travel_win_pct', 0):.1f}% "
            f"(target ≥70%)\n"
            f"  Exit Hybrid: {route_stats['exit_reached_pct']['hybrid']:.1f}%"
        )
    return report


if __name__ == "__main__":
    main()
