#!/usr/bin/env python3
"""
Substantial retrain for QuantumRelief hackathon demo.

1) Regenerate a large Manila dynamic-routing dataset
2) Train Classical FiLM longer (ablation / hybrid seed)
3) Train Hybrid QML (PennyLane PHN) for real — save film_hybrid.pt
4) Smoke-test EXIT REACHED % vs Dijkstra on held-out scenarios
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
from src.film_model import train_film_model
from src.graph_setup import load_or_build_graph, random_epicenter, select_exit_nodes
from src.quantum_hybrid import (
    estimate_quantum_contribution_pct,
    load_hybrid_model,
    quantum_status,
    train_hybrid_model,
)
from src.utils import DATASET_PATH, HYBRID_CHECKPOINT, MODEL_CHECKPOINT, ensure_dirs


def _random_start(G, exits, rng):
    nodes = [n for n in G.nodes() if n not in exits]
    return nodes[int(rng.integers(0, len(nodes)))]


def eval_routes(n_trials: int = 24, seed: int = 7) -> dict:
    """Roll out Hybrid vs Dijkstra; report EXIT REACHED and hop/time stats."""
    # Import after path setup — predict_route lives in app
    from app import dijkstra_route, predict_route, route_overlap_accuracy

    G = load_or_build_graph()
    exits = select_exit_nodes(G, n_exits=3, seed=42)
    model = load_hybrid_model()
    ds = load_dataset()
    mean, std = ds["mean"], ds["std"]
    rng = np.random.default_rng(seed)

    reached = 0
    assist = 0
    hops_h, hops_d = [], []
    time_h, time_d = [], []
    overlaps = []

    for i in range(n_trials):
        dest = exits[int(rng.integers(0, len(exits)))]
        start = _random_start(G, exits, rng)
        epi_ll, _ = random_epicenter(G, seed=int(rng.integers(0, 1_000_000)))
        try:
            path, _, _, travel, _, meta = predict_route(
                G, model, mean, std, start, dest, epi_ll
            )
            dpath, dtravel = dijkstra_route(G, start, dest, epi_ll)
        except Exception as exc:
            print(f"  trial {i}: skip ({exc})")
            continue
        ok = bool(meta.get("reached")) and path[-1] == dest
        reached += int(ok)
        assist += int(bool(meta.get("dijkstra_assist")))
        hops_h.append(max(0, len(path) - 1))
        hops_d.append(max(0, len(dpath) - 1) if dpath else 0)
        time_h.append(float(travel))
        time_d.append(float(dtravel))
        if dpath:
            overlaps.append(route_overlap_accuracy(path, dpath))
        print(
            f"  trial {i:02d}: exit={'YES' if ok else 'NO'}  "
            f"hops={len(path)-1}/{len(dpath)-1 if dpath else '-'}  "
            f"t={travel:.1f}/{dtravel:.1f}  "
            f"assist={meta.get('assist_hops', 0)}  "
            f"reason={meta.get('assist_reason')}"
        )

    n = max(len(hops_h), 1)
    return {
        "n_trials": len(hops_h),
        "exit_reached_pct": 100.0 * reached / n,
        "assist_pct": 100.0 * assist / n,
        "mean_hops_hybrid": float(np.mean(hops_h)) if hops_h else 0.0,
        "mean_hops_dijkstra": float(np.mean(hops_d)) if hops_d else 0.0,
        "mean_time_hybrid": float(np.mean(time_h)) if time_h else 0.0,
        "mean_time_dijkstra": float(np.mean(time_d)) if time_d else 0.0,
        "mean_overlap_pct": float(np.mean(overlaps)) if overlaps else 0.0,
        "quantum_contrib_pct": estimate_quantum_contribution_pct(model),
    }


def main():
    ensure_dirs()
    t0 = time.time()
    print("=== QuantumRelief hackathon retrain ===")
    print(json.dumps(quantum_status(), indent=2))

    # Larger dataset: ~400 episodes → typically 4k–8k decision samples
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    classical_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    # Hybrid is slow (PennyLane per-sample); still train properly
    hybrid_q_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    hybrid_ft_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 6

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
        f"(phase A={hybrid_q_epochs}, B={hybrid_ft_epochs})…"
    )
    if HYBRID_CHECKPOINT.exists():
        HYBRID_CHECKPOINT.unlink()
    # Cap samples for hybrid wall-clock: use full set if small, else up to 4k
    n = len(ds["y"])
    if n > 4000:
        rng = np.random.default_rng(1)
        take = rng.choice(n, size=4000, replace=False)
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

    print("\n[4/4] Route smoke-test (Hybrid vs Dijkstra)…")
    route_stats = eval_routes(n_trials=24, seed=7)
    print(json.dumps(route_stats, indent=2))

    report = {
        "n_samples": int(len(ds["y"])),
        "n_episodes": n_episodes,
        "classical": classical_metrics,
        "hybrid": hybrid_metrics,
        "routes": route_stats,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoints": {
            "classical": str(MODEL_CHECKPOINT),
            "hybrid": str(HYBRID_CHECKPOINT),
            "dataset": str(DATASET_PATH),
        },
    }
    out = ROOT / "data" / "retrain_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Done in {report['elapsed_sec']}s")
    return report


if __name__ == "__main__":
    main()
