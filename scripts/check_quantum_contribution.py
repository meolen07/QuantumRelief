#!/usr/bin/env python3
"""Tiny unit check: load Hybrid checkpoint and print quantum contribution %."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quantum_hybrid import (  # noqa: E402
    estimate_quantum_contribution_pct,
    load_hybrid_model,
)


def main() -> int:
    # Prefer prev while an active retrain may be rewriting film_hybrid.pt
    candidates = [
        ROOT / "models" / "film_hybrid_prev.pt",
        ROOT / "models" / "film_hybrid.pt",
        ROOT / "models" / "film_hybrid_partial.pt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("FAIL: no hybrid checkpoint found")
        return 1

    model = load_hybrid_model(path)
    pct = estimate_quantum_contribution_pct(model)
    print(f"checkpoint={path.name}")
    print(f"estimate_quantum_contribution_pct={pct}")

    if pct is None or pct <= 0:
        print("FAIL: expected positive PHN contribution")
        return 1

    # Streamlit @st.cache_resource keeps instances across module reloads —
    # isinstance(model, HybridFiLMNetwork) would fail; duck-typing must not.
    qh = importlib.import_module("src.quantum_hybrid")
    qh = importlib.reload(qh)
    pct2 = qh.estimate_quantum_contribution_pct(model)
    print(f"after_module_reload={pct2}")
    if pct2 is None or abs(float(pct2) - float(pct)) > 0.5:
        print("FAIL: reload broke contribution estimate (isinstance trap)")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
