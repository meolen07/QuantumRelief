#!/usr/bin/env python3
"""
Honest head-to-head: Hybrid QML vs Classical FiLM vs Dijkstra.

Freeze a Hybrid checkpoint (default: models/film_hybrid_eval_freeze.pt or
film_hybrid_prev.pt) so a concurrent retrain writing film_hybrid.pt cannot
race the eval. Never forges Classical failures.

Usage:
  .venv/bin/python -u scripts/proof_hybrid_vs_classical.py
  .venv/bin/python -u scripts/proof_hybrid_vs_classical.py --n-trials 48 --seed 7
  .venv/bin/python -u scripts/proof_hybrid_vs_classical.py --hybrid models/film_hybrid_prev.pt

Writes: data/proof_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_generation import load_dataset
from src.film_model import load_film_model
from src.graph_setup import load_or_build_graph, random_epicenter, select_exit_nodes
from src.quantum_hybrid import estimate_quantum_contribution_pct, load_hybrid_model
from src.routing_service import compare_three_way
from src.utils import DATA_DIR, MODELS_DIR, ensure_dirs, node_xy, project_local_km, euclidean

HARD_SEEDS_PATH = DATA_DIR / "hard_seeds.json"
DEMO_PATH = DATA_DIR / "demo_scenarios.json"
OUT_PATH = DATA_DIR / "proof_eval_report.json"


def _resolve_hybrid_ckpt(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for name in (
        "film_hybrid_eval_freeze.pt",
        "film_hybrid_prev.pt",
        "film_hybrid.pt",
    ):
        p = MODELS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("No hybrid checkpoint found under models/")


def _path_min_epi_km(G, path: Sequence, epi_ll: Tuple[float, float]) -> float:
    """Safety proxy: minimum km distance of any path node to epicenter."""
    if not path:
        return float("nan")
    origin = node_xy(G, path[0])
    epi_km = project_local_km(float(epi_ll[0]), float(epi_ll[1]), origin[0], origin[1])
    dists = []
    for n in path:
        xy = node_xy(G, n)
        nk = project_local_km(xy[0], xy[1], origin[0], origin[1])
        dists.append(euclidean(nk, epi_km))
    return float(min(dists)) if dists else float("nan")


def _load_hard_trials(hard_repeats: int = 2) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    if not HARD_SEEDS_PATH.exists():
        return trials
    payload = json.loads(HARD_SEEDS_PATH.read_text(encoding="utf-8"))
    for seed in payload.get("seeds") or []:
        for _ in range(hard_repeats):
            trials.append(
                {
                    "kind": "hard",
                    "id": seed.get("id"),
                    "start": seed["start_node"],
                    "dest": seed["dest_node"],
                    "epi_ll": (float(seed["epi_lon"]), float(seed["epi_lat"])),
                }
            )
    return trials


def _build_trials(n_trials: int, seed: int, hard_repeats: int) -> List[Dict[str, Any]]:
    G = load_or_build_graph()
    exits = select_exit_nodes(G, n_exits=3, seed=42)
    start_pool = [n for n in G.nodes() if n not in exits]
    rng = np.random.default_rng(seed)
    trials = _load_hard_trials(hard_repeats=hard_repeats)
    while len(trials) < n_trials:
        start = start_pool[int(rng.integers(0, len(start_pool)))]
        dest = exits[int(rng.integers(0, len(exits)))]
        epi_ll, _ = random_epicenter(G, seed=int(rng.integers(0, 1_000_000)))
        trials.append(
            {
                "kind": "random",
                "id": None,
                "start": start,
                "dest": dest,
                "epi_ll": epi_ll,
            }
        )
    rng.shuffle(trials)
    return trials[:n_trials]


def _eval_one(
    G,
    hybrid,
    classical,
    mean,
    std,
    trial: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
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
        return {"error": str(exc), **{k: trial.get(k) for k in ("kind", "id")}}

    h, c, d = cmp["hybrid"], cmp["classical"], cmp["dijkstra"]
    narr = cmp.get("narrative") or {}
    row = {
        "kind": trial.get("kind"),
        "id": trial.get("id"),
        "start_node": int(start) if str(start).isdigit() else start,
        "dest_node": int(dest) if str(dest).isdigit() else dest,
        "epi_lon": float(epi_ll[0]),
        "epi_lat": float(epi_ll[1]),
        "hybrid_time": float(h["travel_time"]),
        "classical_time": float(c["travel_time"]) if c else None,
        "dijkstra_time": float(d["travel_time"]) if d else None,
        "hybrid_exit": bool(h["exit_reached"]),
        "classical_exit": bool(c["exit_reached"]) if c else False,
        "dijkstra_exit": bool(d["exit_reached"]) if d else False,
        "hybrid_overlap_pct": float(h.get("overlap_vs_dijkstra_pct") or 0.0),
        "classical_overlap_pct": float(c.get("overlap_vs_dijkstra_pct") or 0.0)
        if c
        else None,
        "hybrid_hops": int(h["hops"]),
        "classical_hops": int(c["hops"]) if c else None,
        "dijkstra_hops": int(d["hops"]) if d else None,
        "paths_diverge": bool(narr.get("paths_diverge")),
        "hybrid_beats_classical": bool(narr.get("hybrid_beats_classical")),
        "hybrid_near_dijkstra": bool(narr.get("hybrid_near_dijkstra")),
        "assist": bool(h["meta"].get("dijkstra_assist")),
        "hybrid_min_epi_km": _path_min_epi_km(G, h["path"], epi_ll),
        "classical_min_epi_km": (
            _path_min_epi_km(G, c["path"], epi_ll) if c else float("nan")
        ),
        "dijkstra_min_epi_km": (
            _path_min_epi_km(G, d["path"], epi_ll) if d else float("nan")
        ),
    }
    return row


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    compared = [
        r
        for r in ok
        if r.get("classical_time") is not None
        and r["hybrid_exit"]
        and r["classical_exit"]
    ]
    hard = [r for r in compared if r.get("kind") == "hard"]

    def mean_key(xs: List[Dict], key: str) -> float:
        vals = [float(r[key]) for r in xs if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    n_cmp = max(len(compared), 1)
    travel_wins = sum(
        1 for r in compared if r["hybrid_time"] <= r["classical_time"] * 1.001
    )
    overlap_wins = sum(
        1
        for r in compared
        if r["hybrid_overlap_pct"] + 1e-9 >= float(r["classical_overlap_pct"] or 0.0)
    )
    safety_wins = sum(
        1
        for r in compared
        if np.isfinite(r["hybrid_min_epi_km"])
        and np.isfinite(r["classical_min_epi_km"])
        and r["hybrid_min_epi_km"] + 1e-9 >= r["classical_min_epi_km"]
    )
    hard_travel_wins = sum(
        1 for r in hard if r["hybrid_time"] <= r["classical_time"] * 1.001
    )
    n_hard = max(len(hard), 1)

    mt_h = mean_key(ok, "hybrid_time")
    mt_c = mean_key(ok, "classical_time")
    mt_d = mean_key(ok, "dijkstra_time")

    return {
        "n_trials": len(ok),
        "n_errors": sum(1 for r in rows if "error" in r),
        "n_compared": len(compared),
        "n_hard_compared": len(hard),
        "exit_reached_pct": {
            "hybrid": 100.0 * sum(r["hybrid_exit"] for r in ok) / max(len(ok), 1),
            "classical": 100.0
            * sum(r["classical_exit"] for r in ok)
            / max(len(ok), 1),
            "dijkstra": 100.0
            * sum(r["dijkstra_exit"] for r in ok)
            / max(len(ok), 1),
        },
        "assist_pct": 100.0 * sum(r["assist"] for r in ok) / max(len(ok), 1),
        "mean_time": {"hybrid": mt_h, "classical": mt_c, "dijkstra": mt_d},
        "mean_overlap_pct": {
            "hybrid": mean_key(ok, "hybrid_overlap_pct"),
            "classical": mean_key(ok, "classical_overlap_pct"),
        },
        "mean_min_epi_km": {
            "hybrid": mean_key(ok, "hybrid_min_epi_km"),
            "classical": mean_key(ok, "classical_min_epi_km"),
            "dijkstra": mean_key(ok, "dijkstra_min_epi_km"),
        },
        "delta_mean_travel_vs_classical": float(mt_h - mt_c)
        if np.isfinite(mt_h) and np.isfinite(mt_c)
        else None,
        "hybrid_travel_win_pct": 100.0 * travel_wins / n_cmp,
        "hybrid_overlap_win_pct": 100.0 * overlap_wins / n_cmp,
        "hybrid_safer_min_epi_pct": 100.0 * safety_wins / n_cmp,
        "hybrid_beats_classical_pct": 100.0
        * sum(1 for r in ok if r.get("hybrid_beats_classical"))
        / max(len(ok), 1),
        "hybrid_near_dijkstra_pct": 100.0
        * sum(1 for r in ok if r.get("hybrid_near_dijkstra"))
        / max(len(ok), 1),
        "hard_only": {
            "n": len(hard),
            "mean_time": {
                "hybrid": mean_key(hard, "hybrid_time"),
                "classical": mean_key(hard, "classical_time"),
                "dijkstra": mean_key(hard, "dijkstra_time"),
            },
            "mean_overlap_pct": {
                "hybrid": mean_key(hard, "hybrid_overlap_pct"),
                "classical": mean_key(hard, "classical_overlap_pct"),
            },
            "hybrid_travel_win_pct": 100.0 * hard_travel_wins / n_hard,
        },
        "goal_check": {
            "hybrid_mean_travel_le_classical": bool(
                np.isfinite(mt_h) and np.isfinite(mt_c) and mt_h <= mt_c
            ),
            "hybrid_overlap_ge_classical": bool(
                mean_key(ok, "hybrid_overlap_pct")
                >= mean_key(ok, "classical_overlap_pct")
            ),
            "hybrid_safer_mean_min_epi": bool(
                mean_key(ok, "hybrid_min_epi_km")
                >= mean_key(ok, "classical_min_epi_km")
            ),
        },
    }


def _advantage_candidates(rows: List[Dict[str, Any]], top_n: int = 5) -> List[Dict]:
    """Hybrid near Dijkstra + Classical clearly worse on time or overlap."""
    hits = []
    for r in rows:
        if "error" in r or not r.get("hybrid_exit") or not r.get("classical_exit"):
            continue
        if not r.get("hybrid_near_dijkstra"):
            continue
        ht, ct, dt = r["hybrid_time"], r["classical_time"], r["dijkstra_time"]
        if ct is None or dt is None:
            continue
        ov_h, ov_c = r["hybrid_overlap_pct"], float(r["classical_overlap_pct"] or 0)
        classical_worse = (
            ct > ht * 1.05
            or ov_c < ov_h - 5.0
            or abs(ct - dt) > abs(ht - dt) * 1.05 + 1e-6
        )
        if not classical_worse and not r.get("paths_diverge"):
            continue
        time_gap = max(0.0, ct / max(ht, 1e-6) - 1.0)
        score = (
            3.0 * time_gap
            + 2.5 * max(0.0, (ov_h - ov_c) / 100.0)
            + (2.0 if r.get("paths_diverge") else 0.0)
            + max(0.0, 1.25 - ht / max(dt, 1e-6))
        )
        hits.append({**r, "advantage_score": float(round(score, 4))})
    hits.sort(key=lambda x: x["advantage_score"], reverse=True)
    # Dedupe by (start, dest, epi rounded)
    seen = set()
    out = []
    for h in hits:
        key = (
            h["start_node"],
            h["dest_node"],
            round(h["epi_lat"], 5),
            round(h["epi_lon"], 5),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= top_n:
            break
    return out


def _verify_demo_scenarios(G, hybrid, classical, mean, std) -> List[Dict[str, Any]]:
    if not DEMO_PATH.exists():
        return []
    payload = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
    results = []
    for s in payload.get("scenarios") or []:
        trial = {
            "kind": "demo",
            "id": s.get("id"),
            "start": s["start_node"],
            "dest": s["dest_node"],
            "epi_ll": (float(s["epi_lon"]), float(s["epi_lat"])),
        }
        row = _eval_one(G, hybrid, classical, mean, std, trial)
        if row and "error" not in row:
            expected = (s.get("metrics") or {}).get("classical_time")
            row["expected_classical_time"] = expected
            row["title"] = s.get("title")
            row["still_advantage"] = bool(
                row["hybrid_near_dijkstra"]
                and row["classical_time"] is not None
                and (
                    row["classical_time"] > row["hybrid_time"] * 1.05
                    or row["paths_diverge"]
                )
            )
        results.append(row)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hard-repeats", type=int, default=2)
    ap.add_argument("--hybrid", type=str, default=None)
    ap.add_argument("--classical", type=str, default=None)
    args = ap.parse_args()

    ensure_dirs()
    hybrid_path = _resolve_hybrid_ckpt(args.hybrid)
    classical_path = (
        Path(args.classical)
        if args.classical
        else MODELS_DIR / "film_classical.pt"
    )
    if not classical_path.is_absolute():
        classical_path = ROOT / classical_path

    t0 = time.perf_counter()
    print(f"[proof] hybrid={hybrid_path.name}  classical={classical_path.name}")
    G = load_or_build_graph()
    hybrid = load_hybrid_model(hybrid_path)
    classical = load_film_model(classical_path)
    ds = load_dataset()
    mean, std = ds["mean"], ds["std"]
    q_contrib = estimate_quantum_contribution_pct(hybrid)
    print(f"[proof] quantum_contrib≈{q_contrib:.1f}%")

    # Val accuracy from dataset (next-hop CE labels) if present
    val_acc = {}
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.tensor(ds["X"], dtype=torch.float32)
        y = torch.tensor(ds["y"], dtype=torch.long)
        # Holdout last 15% for a quick next-hop accuracy probe (not train split)
        n = len(y)
        cut = int(n * 0.85)
        Xv, yv = X[cut:], y[cut:]
        loader = DataLoader(TensorDataset(Xv, yv), batch_size=256, shuffle=False)

        def _acc(model) -> float:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in loader:
                    logits = model(xb)
                    pred = logits.argmax(dim=-1)
                    correct += int((pred == yb).sum().item())
                    total += int(yb.numel())
            return 100.0 * correct / max(total, 1)

        val_acc = {
            "hybrid_next_hop_acc_pct": _acc(hybrid),
            "classical_next_hop_acc_pct": _acc(classical),
            "n_holdout": int(len(yv)),
            "note": "Last 15% of routing_dataset.npz — not the exact retrain val split",
        }
        print(
            f"[proof] next-hop holdout acc  H={val_acc['hybrid_next_hop_acc_pct']:.1f}%  "
            f"C={val_acc['classical_next_hop_acc_pct']:.1f}%  (n={val_acc['n_holdout']})"
        )
    except Exception as exc:
        print(f"[proof] next-hop acc skipped ({exc})")
        val_acc = {"error": str(exc)}

    trials = _build_trials(args.n_trials, args.seed, args.hard_repeats)
    print(
        f"[proof] rolling out {len(trials)} trials "
        f"(hard≈{sum(1 for t in trials if t['kind']=='hard')}, seed={args.seed})…"
    )
    rows: List[Dict[str, Any]] = []
    for i, trial in enumerate(trials):
        row = _eval_one(G, hybrid, classical, mean, std, trial)
        if row is None:
            continue
        rows.append(row)
        if "error" in row:
            print(f"  {i:02d} ERROR {row['error']}")
            continue
        print(
            f"  {i:02d} [{row['kind']}{(':'+str(row['id'])) if row.get('id') else ''}] "
            f"H={row['hybrid_time']:.1f} C={row['classical_time']:.1f} "
            f"D={row['dijkstra_time']:.1f}  "
            f"ovH={row['hybrid_overlap_pct']:.0f} ovC={row['classical_overlap_pct']:.0f}  "
            f"minEpi H/C={row['hybrid_min_epi_km']:.2f}/{row['classical_min_epi_km']:.2f}"
        )

    summary = _summarize(rows)
    advantages = _advantage_candidates(rows, top_n=5)
    print("\n[proof] verifying curated demo_scenarios.json…")
    demo = _verify_demo_scenarios(G, hybrid, classical, mean, std)
    for d in demo:
        if not d or "error" in d:
            print(f"  demo {d}: skip")
            continue
        flag = "ADV" if d.get("still_advantage") else "—"
        print(
            f"  {d.get('id')} {flag}  H={d['hybrid_time']:.1f} "
            f"C={d['classical_time']:.1f} D={d['dijkstra_time']:.1f}  "
            f"nearD={d['hybrid_near_dijkstra']} diverge={d['paths_diverge']}"
        )

    # Verdict (scientifically honest)
    g = summary["goal_check"]
    travel_win = summary["hybrid_travel_win_pct"]
    mean_ok = g["hybrid_mean_travel_le_classical"]
    hard_win = summary["hard_only"]["hybrid_travel_win_pct"]
    demo_adv = sum(1 for d in demo if d and d.get("still_advantage"))
    if mean_ok and travel_win >= 55 and g["hybrid_overlap_ge_classical"]:
        verdict = "better"
        verdict_detail = (
            "Hybrid wins on mean travel, pairwise travel win-rate, and mean Dijkstra overlap."
        )
    elif travel_win >= 60 or hard_win >= 70 or demo_adv >= 2:
        verdict = "mixed"
        verdict_detail = (
            "Hybrid wins often pairwise (esp. hard/demo corridors) but does not "
            "dominate all aggregate means — do not claim unconditional superiority."
        )
    else:
        verdict = "not_yet"
        verdict_detail = (
            "Current frozen checkpoint does not show clear Hybrid superiority on this eval."
        )

    elapsed = float(round(time.perf_counter() - t0, 1))
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "methodology": {
            "n_trials": args.n_trials,
            "seed": args.seed,
            "hard_repeats": args.hard_repeats,
            "trial_mix": "hard_seeds.json × hard_repeats + random start/exit/epi",
            "hybrid_checkpoint": str(hybrid_path),
            "classical_checkpoint": str(classical_path),
            "dataset": str(DATA_DIR / "routing_dataset.npz"),
            "travel_win_rule": "hybrid_time <= classical_time * 1.001",
            "near_dijkstra_rule": "hybrid_time <= dijkstra_time * 1.25",
            "safety_proxy": "min km distance of path nodes to epicenter (higher=safer)",
            "honest": "No forged Classical failures; times are path sums under Algorithm-1 dynamics",
        },
        "quantum_contrib_pct": float(q_contrib) if q_contrib is not None else None,
        "next_hop_holdout": val_acc,
        "aggregate": summary,
        "advantage_scenarios": [
            {
                "id": a.get("id") or f"eval_{i+1}",
                "kind": a.get("kind"),
                "start_node": a["start_node"],
                "dest_node": a["dest_node"],
                "epi_lat": a["epi_lat"],
                "epi_lon": a["epi_lon"],
                "hybrid_time": a["hybrid_time"],
                "classical_time": a["classical_time"],
                "dijkstra_time": a["dijkstra_time"],
                "hybrid_overlap_pct": a["hybrid_overlap_pct"],
                "classical_overlap_pct": a["classical_overlap_pct"],
                "paths_diverge": a["paths_diverge"],
                "advantage_score": a["advantage_score"],
                "hybrid_min_epi_km": a["hybrid_min_epi_km"],
                "classical_min_epi_km": a["classical_min_epi_km"],
            }
            for i, a in enumerate(advantages)
        ],
        "demo_scenario_recheck": [
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "hybrid_time": d.get("hybrid_time"),
                "classical_time": d.get("classical_time"),
                "dijkstra_time": d.get("dijkstra_time"),
                "still_advantage": d.get("still_advantage"),
                "paths_diverge": d.get("paths_diverge"),
                "hybrid_near_dijkstra": d.get("hybrid_near_dijkstra"),
            }
            for d in demo
            if d and "error" not in d
        ],
        "prior_retrain_report": None,
        "elapsed_sec": elapsed,
        "reproduce": [
            f".venv/bin/python -u scripts/proof_hybrid_vs_classical.py --n-trials {args.n_trials} --seed {args.seed}",
            ".venv/bin/python -u scripts/find_advantage_scenarios.py 80 5 42",
            ".venv/bin/python -u scripts/check_quantum_contribution.py",
        ],
    }
    prior = DATA_DIR / "retrain_report.json"
    if prior.exists():
        try:
            report["prior_retrain_report"] = json.loads(prior.read_text(encoding="utf-8"))
        except Exception:
            pass

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[proof] VERDICT={verdict}")
    print(f"  {verdict_detail}")
    print(
        f"  mean travel H/C/D={summary['mean_time']['hybrid']:.3f}/"
        f"{summary['mean_time']['classical']:.3f}/{summary['mean_time']['dijkstra']:.3f}"
    )
    print(
        f"  travel_win={summary['hybrid_travel_win_pct']:.1f}%  "
        f"overlap_win={summary['hybrid_overlap_win_pct']:.1f}%  "
        f"near_dij={summary['hybrid_near_dijkstra_pct']:.1f}%  "
        f"hard_travel_win={summary['hard_only']['hybrid_travel_win_pct']:.1f}%"
    )
    print(f"  advantage hits listed: {len(advantages)}  demo still ADV: {demo_adv}/{len(demo)}")
    print(f"[proof] wrote {OUT_PATH} ({elapsed}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
