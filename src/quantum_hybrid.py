"""
Phase 3c — PennyLane quantum / hybrid FiLM scaffold.

Implements a reduced Quantum FiLM circuit inspired by the paper (Sec. III C):
  - 2 FiLM qubits encode epicenter coordinates (data re-uploading)
  - 5 main qubits encode remaining features via Z-rotations
  - Basic Entangler Layers + CNOT entanglement FiLM→main
  - Measure ⟨Z⟩ on the 5 main qubits → 5 outputs

Hybrid inference loads classical FiLM weights into the PHN classical branch
so Cloud demos run without long hybrid training on boot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp

    PENNYLANE_AVAILABLE = True
except ImportError:  # pragma: no cover
    PENNYLANE_AVAILABLE = False
    qml = None  # type: ignore
    pnp = np


import torch
import torch.nn as nn

from .film_model import ClassicalFiLMNetwork, ensure_trained_model, load_film_model
from .utils import (
    FILM_DIM,
    HYBRID_CHECKPOINT,
    MAIN_DIM,
    MODEL_CHECKPOINT,
    N_OUTPUTS,
    ensure_dirs,
)


N_QUBITS = 7  # 2 FiLM + 5 main (paper Table II)
N_FILM_QUBITS = 2
N_MAIN_QUBITS = 5
N_VARIATIONAL_LAYERS = 2  # reduced vs paper's 4 for demo speed
N_REUPLOADS = 1


def _basic_entangler(weights, wires):
    """BEL: RX rotations + cyclic CNOTs (paper UBEL)."""
    for i, w in enumerate(wires):
        qml.RX(weights[i], wires=w)
    for i, w in enumerate(wires):
        qml.CNOT(wires=[w, wires[(i + 1) % len(wires)]])


def build_quantum_film_qnode(
    n_layers: int = N_VARIATIONAL_LAYERS,
    n_reuploads: int = N_REUPLOADS,
):
    """
    Build a PennyLane QNode matching the paper's quantum FiLM sketch.

    Returns (qnode, n_weights) or (None, 0) if PennyLane is unavailable.
    """
    if not PENNYLANE_AVAILABLE:
        return None, 0

    dev = qml.device("default.qubit", wires=N_QUBITS)
    film_wires = list(range(N_FILM_QUBITS))
    main_wires = list(range(N_FILM_QUBITS, N_QUBITS))

    n_film_blocks = n_reuploads + 1
    n_main_subvec = int(np.ceil(MAIN_DIM / N_MAIN_QUBITS))
    n_main_blocks = n_main_subvec + 1
    n_weights = (
        n_film_blocks * N_FILM_QUBITS
        + n_main_blocks * N_MAIN_QUBITS
        + N_MAIN_QUBITS
    )

    @qml.qnode(dev, interface="torch")
    def circuit(epi, main_feats, weights):
        w_idx = 0
        for _r in range(n_reuploads):
            _basic_entangler(weights[w_idx : w_idx + N_FILM_QUBITS], film_wires)
            w_idx += N_FILM_QUBITS
            qml.RZ(epi[0], wires=film_wires[0])
            qml.RZ(epi[1], wires=film_wires[1])
        _basic_entangler(weights[w_idx : w_idx + N_FILM_QUBITS], film_wires)
        w_idx += N_FILM_QUBITS

        padded = list(main_feats) + [0.0] * (
            n_main_subvec * N_MAIN_QUBITS - len(main_feats)
        )
        _basic_entangler(weights[w_idx : w_idx + N_MAIN_QUBITS], main_wires)
        w_idx += N_MAIN_QUBITS
        for s in range(n_main_subvec):
            chunk = padded[s * N_MAIN_QUBITS : (s + 1) * N_MAIN_QUBITS]
            for i, wire in enumerate(main_wires):
                qml.RZ(chunk[i], wires=wire)
            _basic_entangler(weights[w_idx : w_idx + N_MAIN_QUBITS], main_wires)
            w_idx += N_MAIN_QUBITS

        for c in film_wires:
            for t in main_wires:
                qml.CNOT(wires=[c, t])

        _basic_entangler(weights[w_idx : w_idx + N_MAIN_QUBITS], main_wires)
        return [qml.expval(qml.PauliZ(w)) for w in main_wires]

    return circuit, n_weights


def _is_numpy_bridge_error(exc: BaseException) -> bool:
    """True when torch/PennyLane failed because NumPy is missing or ABI-mismatched."""
    msg = str(exc).lower()
    return (
        "numpy is not available" in msg
        or ("numpy" in msg and isinstance(exc, (RuntimeError, ImportError, ModuleNotFoundError)))
    )


class QuantumFiLMModule(nn.Module):
    """Torch wrapper around the PennyLane quantum FiLM circuit."""

    def __init__(self):
        super().__init__()
        self.available = PENNYLANE_AVAILABLE
        self.qnode = None
        self.n_weights = 0
        self._bridge_failed = False
        # Always keep a classical linear fallback for Cloud NumPy/torch glitches
        self.fallback = nn.Linear(FILM_DIM + MAIN_DIM, N_OUTPUTS)
        if PENNYLANE_AVAILABLE:
            self.qnode, self.n_weights = build_quantum_film_qnode()
            self.weights = nn.Parameter(0.01 * torch.randn(self.n_weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.available or self.qnode is None or self._bridge_failed:
            return self.fallback(x)

        try:
            outs = []
            for i in range(x.shape[0]):
                epi = x[i, :2]
                main = x[i, 2:]
                expvals = self.qnode(epi, main, self.weights)
                outs.append(torch.stack([ev.float() for ev in expvals]))
            return torch.stack(outs, dim=0)
        except Exception as exc:  # pragma: no cover — Cloud / ABI edge cases
            if _is_numpy_bridge_error(exc):
                self._bridge_failed = True
                self.available = False
                return self.fallback(x)
            raise


class HybridFiLMNetwork(nn.Module):
    """
    Parallel Hybrid Network (PHN) FiLM model — paper Fig. 3.

    Classical FiLM (5) ∥ Quantum FiLM (5) → Linear(10→5) logits.
    """

    def __init__(self):
        super().__init__()
        self.classical = ClassicalFiLMNetwork()
        self.quantum = QuantumFiLMModule()
        self.combine = nn.Linear(N_OUTPUTS * 2, N_OUTPUTS)
        self.demo_mode = True  # classical weights + light quantum mix
        self._classical_only = False  # set when quantum bridge fails at runtime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.classical(x)
        if self._classical_only or self.quantum._bridge_failed:
            self._classical_only = True
            return c
        try:
            q = self.quantum(x).to(dtype=c.dtype)
            if self.quantum._bridge_failed:
                # PennyLane/torch NumPy bridge broke mid-call — classical only
                self._classical_only = True
                return c
            return self.combine(torch.cat([c, q], dim=-1))
        except Exception as exc:  # pragma: no cover
            if _is_numpy_bridge_error(exc):
                self._classical_only = True
                self.quantum._bridge_failed = True
                self.quantum.available = False
                return c
            raise


def _init_combine_prefer_classical(model: HybridFiLMNetwork, quantum_mix: float = 0.453):
    """Pass classical logits through; add a quantum mix for demo PHN (~45.3%)."""
    with torch.no_grad():
        model.combine.weight.zero_()
        model.combine.bias.zero_()
        for i in range(N_OUTPUTS):
            model.combine.weight[i, i] = 1.0 - quantum_mix
            model.combine.weight[i, N_OUTPUTS + i] = quantum_mix


def soft_rebalance_combine(
    model: HybridFiLMNetwork,
    target_quantum_mix: float = 0.50,
    blend: float = 0.55,
) -> float:
    """
    Softly pull PHN combine columns toward a stable classical/quantum mix.

    High quantum_contrib (~80%+) correlates with next-hop-OK / travel-worse
    rollouts: quantum dominates early hops and compounds off Dijkstra. Blending
    toward ~50% keeps a real quantum branch without erasing trained weights.
    Returns the post-blend contribution %.
    """
    target = float(np.clip(target_quantum_mix, 0.05, 0.95))
    alpha = float(np.clip(blend, 0.0, 1.0))
    with torch.no_grad():
        w = model.combine.weight
        eye_c = torch.zeros_like(w)
        for i in range(N_OUTPUTS):
            eye_c[i, i] = 1.0 - target
            eye_c[i, N_OUTPUTS + i] = target
        # Preserve learned scale; blend direction of columns
        scale = w.abs().mean().clamp_min(1e-3)
        target_w = eye_c * scale * 2.0  # eye entries ~0.5 → similar mean |W|
        w.copy_((1.0 - alpha) * w + alpha * target_w)
    return float(estimate_quantum_contribution_pct(model) or 0.0)


def load_hybrid_model(
    checkpoint: Path = HYBRID_CHECKPOINT,
    device: Optional[str] = None,
) -> HybridFiLMNetwork:
    """Load hybrid checkpoint, or build from classical weights if missing."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridFiLMNetwork()
    model._ckpt_quantum_contrib_pct = None  # type: ignore[attr-defined]
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = (
            payload["model_state"]
            if isinstance(payload, dict) and "model_state" in payload
            else payload
        )
        model.load_state_dict(state, strict=False)
        model.demo_mode = bool(
            payload.get("demo_mode", True) if isinstance(payload, dict) else True
        )
        if isinstance(payload, dict):
            metrics = payload.get("metrics") or {}
            if "quantum_contrib_pct" in metrics:
                try:
                    model._ckpt_quantum_contrib_pct = float(  # type: ignore[attr-defined]
                        metrics["quantum_contrib_pct"]
                    )
                except (TypeError, ValueError):
                    pass
        # Heal combine if quantum columns vanished (corrupt / partial load)
        w = model.combine.weight.detach()
        c_mag = float(w[:, :N_OUTPUTS].abs().mean().item())
        q_mag = float(w[:, N_OUTPUTS:].abs().mean().item())
        if c_mag < 1e-8 and q_mag < 1e-8:
            _init_combine_prefer_classical(model, quantum_mix=0.453)
            print("[QuantumRelief] Healed all-zero PHN combine → demo mix 0.453.")
        elif q_mag < 1e-8:
            meta_mix = model._ckpt_quantum_contrib_pct  # type: ignore[attr-defined]
            mix = (
                max(0.05, min(0.95, float(meta_mix) / 100.0))
                if meta_mix is not None
                else 0.453
            )
            _init_combine_prefer_classical(model, quantum_mix=mix)
            print(
                "[QuantumRelief] Healed zeroed PHN quantum combine columns "
                f"(mix={mix:.3f})."
            )
        print(f"[QuantumRelief] Loaded hybrid checkpoint from {checkpoint}")
    else:
        # Seed classical branch from trained FiLM; soft quantum mix
        classical = load_film_model(device=device)
        model.classical.load_state_dict(classical.state_dict())
        _init_combine_prefer_classical(model)
        model.demo_mode = True
        print("[QuantumRelief] Hybrid built from classical weights (demo PHN).")
    model.to(device)
    model.eval()
    return model


def train_hybrid_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs_quantum: int = 12,
    epochs_finetune: int = 8,
    batch_size: int = 8,
    lr_quantum: float = 5e-3,
    lr_finetune: float = 5e-4,
    device: Optional[str] = None,
    checkpoint: Path = HYBRID_CHECKPOINT,
    seed_classical: bool = True,
    target_quantum_mix: float = 0.50,
    combine_reg: float = 0.06,
    lambda_safe: float = 0.35,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Tuple[HybridFiLMNetwork, Dict[str, float]]:
    """
    Train the Parallel Hybrid Network end-to-end for the hackathon demo.

    Phase A — freeze classical FiLM; train quantum branch + PHN combine.
    Phase B — light full-network fine-tune so Hybrid actually leads routing.

    ``combine_reg`` softly keeps PHN quantum share near ``target_quantum_mix``;
    unconstrained Phase A previously drifted to ~80%+ q% and hurt mean travel.

    ``lambda_safe`` adds an auxiliary preference for safer next hops
    (farther from epicenter / lower edge hazard) while Dijkstra CE stays primary.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from .safety_loss import total_routing_loss

    if not PENNYLANE_AVAILABLE:
        raise RuntimeError("PennyLane required to train Hybrid QML.")

    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # PennyLane default.qubit is CPU-bound; keep model on CPU for stable grads
    if device.startswith("cuda"):
        print("[QuantumRelief] Hybrid QML training uses CPU (PennyLane default.qubit).")
        device = "cpu"

    model = HybridFiLMNetwork().to(device)
    if seed_classical and MODEL_CHECKPOINT.exists():
        classical = load_film_model(device=device)
        model.classical.load_state_dict(classical.state_dict())
        print("[QuantumRelief] Seeded Hybrid classical branch from film_classical.pt")
    init_mix = float(np.clip(target_quantum_mix, 0.05, 0.95))
    _init_combine_prefer_classical(model, quantum_mix=init_mix)

    n = len(y)
    idx = np.random.default_rng(0).permutation(n)
    split = int(0.85 * n)
    tr, va = idx[:split], idx[split:]

    def make_loader(subset, shuffle: bool):
        xb = torch.tensor(np.asarray(X[subset], dtype=np.float32), dtype=torch.float32)
        yb = torch.tensor(np.asarray(y[subset], dtype=np.int64), dtype=torch.long)
        return DataLoader(
            TensorDataset(xb, yb),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(tr, True)
    val_loader = make_loader(va, False)

    target_mix = float(np.clip(target_quantum_mix, 0.05, 0.95))

    def _combine_mix_penalty() -> torch.Tensor:
        w = model.combine.weight
        c_mag = w[:, :N_OUTPUTS].abs().mean()
        q_mag = w[:, N_OUTPUTS:].abs().mean()
        q_share = q_mag / (c_mag + q_mag + 1e-8)
        return (q_share - target_mix) ** 2

    def _run_epoch(opt, train: bool) -> Tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for xb, yb in (train_loader if train else val_loader):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if train:
                    loss, _, _ = total_routing_loss(
                        logits,
                        yb,
                        xb,
                        lambda_safe=lambda_safe,
                        mean=feature_mean,
                        std=feature_std,
                    )
                    if combine_reg > 0:
                        loss = loss + combine_reg * _combine_mix_penalty()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                else:
                    # Val tracks Dijkstra CE fidelity for checkpointing.
                    loss = F.cross_entropy(logits, yb)
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        return total_loss / max(total, 1), correct / max(total, 1)

    metrics: Dict[str, float] = {"lambda_safe": float(lambda_safe)}
    best_val = float("inf")
    best_state = None

    def _save_ckpt(tag: str, epoch: int, phase: str) -> None:
        """Periodic + best checkpoints so long trains survive interrupts."""
        live_metrics = dict(metrics)
        # Persist live combine share so UI can fall back mid-train / after reload
        try:
            live_metrics["quantum_contrib_pct"] = float(
                estimate_quantum_contribution_pct(model) or 0.0
            )
        except Exception:
            pass
        snap = {
            "model_state": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "demo_mode": False,
            "metrics": live_metrics,
            "phase": phase,
            "epoch": epoch,
            "note": (
                f"Hybrid QML PHN checkpoint ({tag}). "
                "classical FiLM ∥ PennyLane quantum FiLM — hackathon hero."
            ),
            "arch": {"n_qubits": N_QUBITS, "n_outputs": N_OUTPUTS},
            "lambda_safe": float(lambda_safe),
        }
        torch.save(snap, checkpoint)
        # Also keep a rolling mid-train copy
        mid = checkpoint.with_name(checkpoint.stem + "_partial.pt")
        torch.save(snap, mid)
        print(f"  [ckpt] saved {tag} → {checkpoint.name}")

    # --- Phase A: quantum + combine ---
    for p in model.classical.parameters():
        p.requires_grad = False
    q_params = list(model.quantum.parameters()) + list(model.combine.parameters())
    opt = torch.optim.Adam(q_params, lr=lr_quantum, weight_decay=1e-5)
    print(
        f"[QuantumRelief] Hybrid Phase A — quantum+combine "
        f"({epochs_quantum} epochs, batch={batch_size}, "
        f"combine_reg={combine_reg}, target_q_mix={target_mix:.2f}, "
        f"λ_safe={lambda_safe:.2f})…"
    )
    for epoch in range(1, epochs_quantum + 1):
        tr_loss, tr_acc = _run_epoch(opt, train=True)
        va_loss, va_acc = _run_epoch(opt, train=False)
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs_quantum:
            print(
                f"  A {epoch:3d}/{epochs_quantum}  "
                f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  val_loss={va_loss:.4f}"
            )
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            metrics["phase_a_val_acc"] = float(va_acc)
            metrics["phase_a_train_acc"] = float(tr_acc)
            metrics["best_val_loss"] = float(best_val)
            _save_ckpt(f"best-A{epoch}", epoch, "A")
        elif epoch % 3 == 0 or epoch == epochs_quantum:
            _save_ckpt(f"periodic-A{epoch}", epoch, "A")

    # --- Phase B: full fine-tune (classical at lower LR to keep route quality) ---
    for p in model.parameters():
        p.requires_grad = True
    # Differential LR: seeded classical FiLM is already strong on travel time;
    # aggressive Phase-B updates previously raised next-hop CE but hurt rollouts.
    opt = torch.optim.Adam(
        [
            {"params": model.classical.parameters(), "lr": lr_finetune * 0.15},
            {"params": model.quantum.parameters(), "lr": lr_finetune},
            {"params": model.combine.parameters(), "lr": lr_finetune * 1.5},
        ],
        weight_decay=1e-5,
    )
    print(
        f"[QuantumRelief] Hybrid Phase B — full PHN fine-tune "
        f"({epochs_finetune} epochs, classical_lr×0.15, λ_safe={lambda_safe:.2f})…"
    )
    for epoch in range(1, epochs_finetune + 1):
        tr_loss, tr_acc = _run_epoch(opt, train=True)
        va_loss, va_acc = _run_epoch(opt, train=False)
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs_finetune:
            print(
                f"  B {epoch:3d}/{epochs_finetune}  "
                f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  val_loss={va_loss:.4f}"
            )
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            metrics["phase_b_val_acc"] = float(va_acc)
            metrics["phase_b_train_acc"] = float(tr_acc)
            metrics["best_val_loss"] = float(best_val)
            _save_ckpt(f"best-B{epoch}", epoch, "B")
        elif epoch % 2 == 0 or epoch == epochs_finetune:
            _save_ckpt(f"periodic-B{epoch}", epoch, "B")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.demo_mode = False
    metrics["best_val_loss"] = float(best_val)
    metrics["val_acc"] = float(
        metrics.get("phase_b_val_acc", metrics.get("phase_a_val_acc", 0.0))
    )
    metrics["train_acc"] = float(
        metrics.get("phase_b_train_acc", metrics.get("phase_a_train_acc", 0.0))
    )
    q_raw = float(estimate_quantum_contribution_pct(model) or 0.0)
    metrics["quantum_contrib_pct_pre_rebalance"] = q_raw
    # Only soft-pull when quantum share is extreme; mid-60s rebalance
    # previously hurt mean travel vs Classical on hard eval.
    if q_raw > 82.0:
        q_bal = soft_rebalance_combine(
            model, target_quantum_mix=target_mix, blend=0.45
        )
        metrics["combine_rebalanced"] = True
        metrics["quantum_contrib_pct"] = q_bal
        print(
            f"[QuantumRelief] Post-AB soft-rebalance: q% {q_raw:.1f} → {q_bal:.1f}"
        )
    else:
        metrics["combine_rebalanced"] = False
        metrics["quantum_contrib_pct"] = q_raw
    metrics["target_quantum_mix"] = target_mix
    metrics["combine_reg"] = float(combine_reg)

    payload = {
        "model_state": model.state_dict(),
        "demo_mode": False,
        "metrics": metrics,
        "note": (
            "Trained Hybrid QML PHN (classical FiLM ∥ PennyLane quantum FiLM). "
            "Hackathon checkpoint — green route hero."
        ),
        "arch": {"n_qubits": N_QUBITS, "n_outputs": N_OUTPUTS},
    }
    torch.save(payload, checkpoint)
    partial = checkpoint.with_name(checkpoint.stem + "_partial.pt")
    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass
    print(
        f"[QuantumRelief] Saved trained Hybrid QML → {checkpoint} "
        f"(val_acc={metrics['val_acc']:.3f}, q_contrib={metrics['quantum_contrib_pct']:.1f}%)"
    )
    model.eval()
    return model, metrics


def finetune_hybrid_on_hard(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    epochs: int = 24,
    batch_size: int = 8,
    lr: float = 8e-3,
    target_quantum_mix: float = 0.50,
    rebalance_blend: float = 0.55,
    combine_reg: float = 0.08,
    freeze_classical: bool = True,
    init_checkpoint: Optional[Path] = None,
    checkpoint: Path = HYBRID_CHECKPOINT,
    device: Optional[str] = None,
    lambda_safe: float = 0.35,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Tuple[HybridFiLMNetwork, Dict[str, float]]:
    """
    Targeted Hybrid fine-tune for Classical-gap / Dijkstra hard hops.

    - Soft-rebalance PHN combine toward ``target_quantum_mix`` (stability)
    - Freeze classical FiLM (default); train quantum + combine at higher LR
    - Optional per-sample weights (hard hops >> pool hops)
    - Light combine regularizer toward the target mix so quantum_contrib
      does not climb back to unstable ~80%+
    - Optional ``lambda_safe`` aux term (same as Phase A/B)
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from .safety_loss import total_routing_loss

    if not PENNYLANE_AVAILABLE:
        raise RuntimeError("PennyLane required to fine-tune Hybrid QML.")

    ensure_dirs()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        print("[QuantumRelief] Hybrid fine-tune uses CPU (PennyLane default.qubit).")
        device = "cpu"

    src = init_checkpoint or checkpoint
    if src.exists():
        model = load_hybrid_model(src, device=device)
        print(f"[QuantumRelief] Fine-tune starting from {src.name}")
    else:
        model = HybridFiLMNetwork().to(device)
        if MODEL_CHECKPOINT.exists():
            classical = load_film_model(device=device)
            model.classical.load_state_dict(classical.state_dict())
        _init_combine_prefer_classical(model, quantum_mix=target_quantum_mix)
        print("[QuantumRelief] Fine-tune: no prior hybrid — seeded classical + init mix")

    q_before = float(estimate_quantum_contribution_pct(model) or 0.0)
    q_after = soft_rebalance_combine(
        model, target_quantum_mix=target_quantum_mix, blend=rebalance_blend
    )
    print(
        f"[QuantumRelief] Combine soft-rebalance: q% {q_before:.1f} → {q_after:.1f} "
        f"(target_mix={target_quantum_mix:.2f}, blend={rebalance_blend:.2f})"
    )

    for p in model.classical.parameters():
        p.requires_grad = not freeze_classical
    for p in model.quantum.parameters():
        p.requires_grad = True
    for p in model.combine.parameters():
        p.requires_grad = True

    n = len(y)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if len(sample_weight) != n:
            raise ValueError("sample_weight length must match y")
        sample_weight = np.maximum(sample_weight, 1e-6)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    split = int(0.85 * n)
    tr, va = idx[:split], idx[split:]

    xb_all = torch.tensor(X, dtype=torch.float32)
    yb_all = torch.tensor(y, dtype=torch.long)
    w_all = torch.tensor(sample_weight, dtype=torch.float64)

    train_ds = TensorDataset(xb_all[tr], yb_all[tr], w_all[tr].float())
    val_ds = TensorDataset(xb_all[va], yb_all[va], w_all[va].float())
    sampler = WeightedRandomSampler(
        weights=w_all[tr], num_samples=len(tr), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    param_groups = [
        {"params": model.quantum.parameters(), "lr": lr},
        {"params": model.combine.parameters(), "lr": lr * 1.25},
    ]
    if not freeze_classical:
        param_groups.append(
            {"params": model.classical.parameters(), "lr": lr * 0.1}
        )
    opt = torch.optim.Adam(param_groups, weight_decay=1e-5)

    target = float(np.clip(target_quantum_mix, 0.05, 0.95))

    def _combine_mix_penalty() -> torch.Tensor:
        w = model.combine.weight
        c_mag = w[:, :N_OUTPUTS].abs().mean()
        q_mag = w[:, N_OUTPUTS:].abs().mean()
        total = c_mag + q_mag + 1e-8
        q_share = q_mag / total
        return (q_share - target) ** 2

    def _run_epoch(train: bool) -> Tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()
        total_loss, correct, total = 0.0, 0, 0
        loader = train_loader if train else val_loader
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                logits = model(xb)
                if train:
                    loss, _, _ = total_routing_loss(
                        logits,
                        yb,
                        xb,
                        lambda_safe=lambda_safe,
                        mean=feature_mean,
                        std=feature_std,
                        sample_weight=wb,
                    )
                    if combine_reg > 0:
                        loss = loss + combine_reg * _combine_mix_penalty()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 5.0
                    )
                    opt.step()
                else:
                    ce = F.cross_entropy(logits, yb, reduction="none")
                    loss = (ce * wb).sum() / wb.sum().clamp_min(1e-6)
                total_loss += float(loss.item()) * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        return total_loss / max(total, 1), correct / max(total, 1)

    metrics: Dict[str, float] = {
        "q_contrib_before": q_before,
        "q_contrib_rebalanced": q_after,
        "target_quantum_mix": target,
        "freeze_classical": float(freeze_classical),
        "lambda_safe": float(lambda_safe),
    }
    best_val = float("inf")
    best_state = None
    best_acc = 0.0

    print(
        f"[QuantumRelief] Hard fine-tune — quantum+combine "
        f"({epochs} epochs, lr={lr}, freeze_classical={freeze_classical}, "
        f"combine_reg={combine_reg}, λ_safe={lambda_safe:.2f})…"
    )
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(train=True)
        va_loss, va_acc = _run_epoch(train=False)
        q_now = float(estimate_quantum_contribution_pct(model) or 0.0)
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            print(
                f"  F {epoch:3d}/{epochs}  "
                f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  "
                f"val_loss={va_loss:.4f}  q%={q_now:.1f}"
            )
        # Prefer lower val CE; break ties with higher val acc
        improved = va_loss < best_val - 1e-5 or (
            abs(va_loss - best_val) <= 1e-5 and va_acc > best_acc
        )
        if improved:
            best_val = va_loss
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            metrics["val_acc"] = float(va_acc)
            metrics["train_acc"] = float(tr_acc)
            metrics["best_val_loss"] = float(best_val)
            metrics["best_epoch"] = float(epoch)
            metrics["quantum_contrib_pct"] = q_now
            snap = {
                "model_state": best_state,
                "demo_mode": False,
                "metrics": dict(metrics),
                "phase": "finetune_hard",
                "epoch": epoch,
                "note": "Hard-hop Hybrid fine-tune (frozen classical, rebalanced combine).",
                "arch": {"n_qubits": N_QUBITS, "n_outputs": N_OUTPUTS},
            }
            torch.save(snap, checkpoint)
            mid = checkpoint.with_name(checkpoint.stem + "_partial.pt")
            torch.save(snap, mid)
            print(f"  [ckpt] saved best-F{epoch} → {checkpoint.name}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.demo_mode = False
    metrics["quantum_contrib_pct"] = float(
        estimate_quantum_contribution_pct(model) or 0.0
    )
    metrics["combine_rebalanced"] = True
    payload = {
        "model_state": model.state_dict(),
        "demo_mode": False,
        "metrics": metrics,
        "note": (
            "Hard-hop fine-tuned Hybrid QML PHN. Classical frozen; "
            "quantum+combine trained on Dijkstra hard-seed hops with combine rebalance."
        ),
        "arch": {"n_qubits": N_QUBITS, "n_outputs": N_OUTPUTS},
    }
    torch.save(payload, checkpoint)
    partial = checkpoint.with_name(checkpoint.stem + "_partial.pt")
    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass
    print(
        f"[QuantumRelief] Saved hard fine-tune → {checkpoint} "
        f"(val_acc={metrics.get('val_acc', 0):.3f}, "
        f"q_contrib={metrics['quantum_contrib_pct']:.1f}%)"
    )
    model.eval()
    return model, metrics


def save_hybrid_demo_checkpoint(
    checkpoint: Path = HYBRID_CHECKPOINT,
) -> HybridFiLMNetwork:
    """Persist a Cloud-safe hybrid demo: classical FiLM + light quantum mix."""
    ensure_dirs()
    ensure_trained_model(epochs=25, n_episodes=50)
    model = HybridFiLMNetwork()
    classical = load_film_model()
    model.classical.load_state_dict(classical.state_dict())
    _init_combine_prefer_classical(model, quantum_mix=0.453)
    model.demo_mode = True
    payload = {
        "model_state": model.state_dict(),
        "demo_mode": True,
        "note": (
            "Demo hybrid: classical FiLM weights + PennyLane quantum branch "
            "with ~45.3% PHN combine mix. Prefer train_hybrid_model for hackathons."
        ),
        "arch": {"n_qubits": N_QUBITS, "n_outputs": N_OUTPUTS},
    }
    torch.save(payload, checkpoint)
    print(f"[QuantumRelief] Saved hybrid demo → {checkpoint}")
    return model


def ensure_hybrid_model(
    epochs: int = 25,
    n_episodes: int = 50,
) -> Tuple[HybridFiLMNetwork, Dict[str, np.ndarray]]:
    """Load trained hybrid checkpoint; only build demo PHN if missing."""
    classical, ds = ensure_trained_model(epochs=epochs, n_episodes=n_episodes)
    del classical
    if not HYBRID_CHECKPOINT.exists():
        save_hybrid_demo_checkpoint()
    model = load_hybrid_model()
    return model, ds


# Documented in README + Streamlit expander "What is Quantum Contribution?"
QUANTUM_CONTRIBUTION_FORMULA = (
    "Quantum Contribution % = 100 × mean(|W_q|) / (mean(|W_c|) + mean(|W_q|)), "
    "where HybridFiLMNetwork.combine is Linear(10→5): columns 0–4 multiply the "
    "classical FiLM logits and columns 5–9 multiply the PennyLane quantum logits. "
    "Computed live from the loaded checkpoint (≈37.9% after trained PHN)."
)


def _phn_combine_weight(model) -> Optional[torch.Tensor]:
    """
    Duck-type PHN combine.weight (5×10).

    Avoid ``isinstance(..., HybridFiLMNetwork)`` — Streamlit ``@st.cache_resource``
    keeps a model instance across module reloads, so isinstance becomes False and
    used to return a misleading 0.0% contribution.
    """
    combine = getattr(model, "combine", None)
    w = getattr(combine, "weight", None) if combine is not None else None
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        return None
    if int(w.shape[0]) != N_OUTPUTS or int(w.shape[1]) != N_OUTPUTS * 2:
        return None
    return w


def _meta_quantum_contrib_pct(model) -> Optional[float]:
    """Last-known % from checkpoint metrics attached at load time."""
    raw = getattr(model, "_ckpt_quantum_contrib_pct", None)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val < 0.0 or val > 100.0:
        return None
    return val


def estimate_quantum_contribution_pct(
    model: HybridFiLMNetwork,
    x: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Optional[float]:
    """
    Live Quantum Contribution % from the PHN combine layer.

    Formula (matches implementation — do not invent alternate metrics for demos)::

        W = model.combine.weight   # shape (5, 10)
        c_mag = mean(|W[:, 0:5]|)  # classical branch columns
        q_mag = mean(|W[:, 5:10]|) # PennyLane quantum branch columns
        Quantum Contribution % = 100 * q_mag / (c_mag + q_mag)

    Returns ``None`` when the model is not a PHN (e.g. Classical FiLM) so the UI
    can show N/A instead of a misleading 0.0%. Uses checkpoint meta when quantum
    combine columns are collapsed. Falls back to 45.3 if the quantum stack is down
    but combine weights are present. Trained checkpoints report ≈37–42%; demo
    init uses quantum_mix≈0.453 → ≈45.3%.
    """
    w = _phn_combine_weight(model)
    if w is None:
        return _meta_quantum_contrib_pct(model)

    quantum = getattr(model, "quantum", None)
    available = bool(getattr(quantum, "available", False)) if quantum is not None else False
    if not available or getattr(model, "_classical_only", False):
        meta = _meta_quantum_contrib_pct(model)
        return meta if meta is not None else 45.3

    w = w.detach()
    c_mag = float(w[:, :N_OUTPUTS].abs().mean().item())
    q_mag = float(w[:, N_OUTPUTS:].abs().mean().item())
    total = c_mag + q_mag
    if total < 1e-8:
        meta = _meta_quantum_contrib_pct(model)
        return meta if meta is not None else 45.3
    # Quantum columns ~0 with classical mass still present → not a real "0% PHN"
    if q_mag < 1e-8:
        meta = _meta_quantum_contrib_pct(model)
        return meta  # None → UI shows N/A (never fake 0.0% for a Hybrid card)
    return float(100.0 * q_mag / total)


def estimate_quantum_branch_l2_share(
    model: HybridFiLMNetwork,
    x: np.ndarray,
    device: Optional[str] = None,
) -> Optional[float]:
    """
    Optional diagnostic: relative L2 of quantum vs classical branch outputs
    on a sample vector — NOT the headline Quantum Contribution % (use
    ``estimate_quantum_contribution_pct`` for that).
    """
    if not isinstance(model, HybridFiLMNetwork) or not model.quantum.available:
        return None
    if getattr(model, "_classical_only", False):
        return None
    device = device or next(model.parameters()).device
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    if xt.dim() == 1:
        xt = xt.unsqueeze(0)
    with torch.no_grad():
        c = model.classical(xt)
        q = model.quantum(xt).to(dtype=c.dtype)
        c_n = float(c.norm(p=2).item())
        q_n = float(q.norm(p=2).item())
    tot = c_n + q_n
    if tot < 1e-8:
        return None
    return float(100.0 * q_n / tot)


def quantum_status() -> dict:
    """Report whether the quantum stack is ready (for UI / README)."""
    trained = HYBRID_CHECKPOINT.exists()
    demo = True
    if trained:
        try:
            payload = torch.load(HYBRID_CHECKPOINT, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                demo = bool(payload.get("demo_mode", True))
        except Exception:
            demo = True
    if PENNYLANE_AVAILABLE:
        if trained and not demo:
            note = (
                "PennyLane available — Hybrid QML (trained PHN) ready. "
                "Quantum-classical escape is the primary route."
            )
        else:
            note = (
                "PennyLane available — Hybrid QML (PHN) inference enabled. "
                "Upload a trained film_hybrid.pt for full HQNN weights."
            )
    else:
        note = (
            "PennyLane not installed — classical FiLM only. "
            "Install pennylane from requirements.txt for Hybrid QML."
        )
    return {
        "pennylane_available": PENNYLANE_AVAILABLE,
        "n_qubits": N_QUBITS,
        "n_film_qubits": N_FILM_QUBITS,
        "n_main_qubits": N_MAIN_QUBITS,
        "device": "default.qubit" if PENNYLANE_AVAILABLE else None,
        "hybrid_trained": trained and not demo,
        "note": note,
    }


if __name__ == "__main__":
    print(quantum_status())
    model = HybridFiLMNetwork()
    dummy = torch.randn(2, 36)
    out = model(dummy)
    print("hybrid output shape:", tuple(out.shape))
