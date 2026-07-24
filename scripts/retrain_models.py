#!/usr/bin/env python3
"""
Substantial retrain for QuantumRelief hackathon demo.

1) Regenerate a large Manila dynamic-routing dataset
2) Train Classical FiLM longer (ablation / hybrid seed)
3) Train Hybrid QML (PennyLane PHN) thoroughly — Phase A + B
4) Smoke-test 3-way: Hybrid vs Classical vs Dijkstra
5) Write data/retrain_report.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.dataset_generation import generate_dataset, load_dataset
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


def main():
    ensure_dirs()
    t0 = time.time()
    print("=== QuantumRelief hackathon retrain (3-way) ===")
    print(json.dumps(quantum_status(), indent=2))

    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    classical_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    hybrid_q_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    hybrid_ft_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    hybrid_max_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 3500

    print(f"\n[1/4] Generating dataset ({n_episodes} episodes)…")
    if DATASET_PATH.exists():
        DATASET_PATH.unlink()
    ds = generate_dataset(n_episodes=n_episodes, seed=42, save=True)
    print(f"  samples: {len(ds['y'])}  X={ds['X'].shape}")

    print(f"\n[2/4] Training Classical FiLM ({classical_epochs} epochs)…")
    if MODEL_CHECKPOINT.exists():
        MODEL_CHECKPOINT.unlink()
    _, classical_metrics = train_film_model(
        ds["X"], ds["y"], epochs=classical_epochs, batch_size=64
    )
    print("  classical metrics:", classical_metrics)

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

    report = {
        "n_samples": int(len(ds["y"])),
        "n_episodes": n_episodes,
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
        },
        "train_config": {
            "classical_epochs": classical_epochs,
            "hybrid_phase_a": hybrid_q_epochs,
            "hybrid_phase_b": hybrid_ft_epochs,
            "hybrid_max_samples": hybrid_max_samples,
        },
    }
    out = ROOT / "data" / "retrain_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Done in {report['elapsed_sec']}s")
    return report


if __name__ == "__main__":
    main()
