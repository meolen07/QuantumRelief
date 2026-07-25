#!/usr/bin/env python3
"""
Substantial retrain for QuantumRelief hackathon demo.

1) (optional --hard) sync Classical-failure seeds + hard hazard labels
2) Regenerate Manila dynamic-routing dataset (default 1000 episodes)
3) Train Classical FiLM (ablation / hybrid seed)
4) Train Hybrid QML (PennyLane PHN) — Phase A + B
5) Smoke-test 3-way: Hybrid vs Classical vs Dijkstra
6) Write data/retrain_report.json

Recommended hard retrain:

  source .venv/bin/activate
  caffeinate -dimsu python -u scripts/retrain_models.py --hard

Or dataset-only:

  python -u -m src.dataset_generation --episodes 1000 --hard
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
    generate_dataset,
    load_dataset,
    load_hard_seeds,
    write_hard_seeds_from_scenarios,
)
from src.film_model import load_film_model, train_film_model
from src.graph_setup import load_or_build_graph, random_epicenter, select_exit_nodes
from src.quantum_hybrid import (
    estimate_quantum_contribution_pct,
    load_hybrid_model,
    quantum_status,
    train_hybrid_model,
)
from src.routing_service import compare_three_way
from src.utils import DATASET_PATH, HYBRID_CHECKPOINT, MODEL_CHECKPOINT, ensure_dirs


def _random_start(G, exits, rng):
    nodes = [n for n in G.nodes() if n not in exits]
    return nodes[int(rng.integers(0, len(nodes)))]


def eval_routes_three_way(n_trials: int = 24, seed: int = 7) -> dict:
    """Roll out Hybrid vs Classical vs Dijkstra; report EXIT + hop/time stats."""
    G = load_or_build_graph()
    exits = select_exit_nodes(G, n_exits=3, seed=42)
    hybrid = load_hybrid_model()
    classical = load_film_model()
    ds = load_dataset()
    mean, std = ds["mean"], ds["std"]
    rng = np.random.default_rng(seed)

    reached_h = reached_c = reached_d = 0
    assist = 0
    time_h, time_c, time_d = [], [], []
    hops_h, hops_c, hops_d = [], [], []
    ov_h, ov_c = [], []
    hybrid_beats_classical = 0
    hybrid_near_dij = 0

    for i in range(n_trials):
        dest = exits[int(rng.integers(0, len(exits)))]
        start = _random_start(G, exits, rng)
        epi_ll, _ = random_epicenter(G, seed=int(rng.integers(0, 1_000_000)))
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
        if cmp["narrative"].get("hybrid_beats_classical"):
            hybrid_beats_classical += 1
        if cmp["narrative"].get("hybrid_near_dijkstra"):
            hybrid_near_dij += 1

        print(
            f"  trial {i:02d}: H={h['travel_time']:.1f}/"
            f"C={c['travel_time'] if c else float('nan'):.1f}/"
            f"D={d['travel_time'] if d else float('nan'):.1f}  "
            f"exit={ok_h}/{ok_c}/{ok_d}  "
            f"ovH={h.get('overlap_vs_dijkstra_pct')}  "
            f"assist={h['meta'].get('assist_hops', 0)}"
        )

    n = max(len(time_h), 1)
    return {
        "n_trials": len(time_h),
        "exit_reached_pct": {
            "hybrid": 100.0 * reached_h / n,
            "classical": 100.0 * reached_c / n,
            "dijkstra": 100.0 * reached_d / n,
        },
        "assist_pct": 100.0 * assist / n,
        "mean_time": {
            "hybrid": float(np.mean(time_h)) if time_h else 0.0,
            "classical": float(np.mean(time_c)) if time_c else 0.0,
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
        "hybrid_beats_classical_pct": 100.0 * hybrid_beats_classical / n,
        "hybrid_near_dijkstra_pct": 100.0 * hybrid_near_dij / n,
        "quantum_contrib_pct": estimate_quantum_contribution_pct(hybrid),
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
    p.add_argument("--hybrid-a", type=int, default=10, help="Hybrid Phase A epochs.")
    p.add_argument("--hybrid-b", type=int, default=6, help="Hybrid Phase B epochs.")
    p.add_argument(
        "--hybrid-max-samples",
        type=int,
        default=3500,
        help="Cap Hybrid train subset size (CPU-bound PennyLane).",
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
        "--dataset-only",
        action="store_true",
        help="Generate/save dataset only; skip training.",
    )
    p.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Train Classical only (still writes partial retrain_report).",
    )
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

    print(f"\n[2/4] Training Classical FiLM ({classical_epochs} epochs)…")
    if MODEL_CHECKPOINT.exists():
        MODEL_CHECKPOINT.unlink()
    _, classical_metrics = train_film_model(
        ds["X"], ds["y"], epochs=classical_epochs, batch_size=64
    )
    print("  classical metrics:", classical_metrics)

    hybrid_metrics = None
    route_stats = None
    if not args.skip_hybrid:
        print(
            f"\n[3/4] Training Hybrid QML PHN "
            f"(phase A={hybrid_q_epochs}, B={hybrid_ft_epochs}, "
            f"max_samples={hybrid_max_samples})…"
        )
        if HYBRID_CHECKPOINT.exists():
            HYBRID_CHECKPOINT.unlink()
        n = len(ds["y"])
        if n > hybrid_max_samples:
            rng = np.random.default_rng(1)
            take = rng.choice(n, size=hybrid_max_samples, replace=False)
            Xh, yh = ds["X"][take], ds["y"][take]
            print(f"  hybrid train subset: {len(yh)} / {n} samples")
        else:
            Xh, yh = ds["X"], ds["y"]
        _, hybrid_metrics = train_hybrid_model(
            Xh,
            yh,
            epochs_quantum=hybrid_q_epochs,
            epochs_finetune=hybrid_ft_epochs,
            batch_size=8,
        )
        print("  hybrid metrics:", hybrid_metrics)

        print("\n[4/4] Route smoke-test (Hybrid vs Classical vs Dijkstra)…")
        route_stats = eval_routes_three_way(n_trials=24, seed=7)
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
            "hybrid_phase_a": hybrid_q_epochs,
            "hybrid_phase_b": hybrid_ft_epochs,
            "hybrid_max_samples": hybrid_max_samples,
            "hard": hard,
            "hazard_intensity": hazard_intensity,
            "oversample": oversample,
        },
    }
    out = ROOT / "data" / "retrain_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Done in {report['elapsed_sec']}s")
    return report


if __name__ == "__main__":
    main()
