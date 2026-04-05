"""Training loop, evaluation, FGSM robustness, checkpoint I/O."""

from __future__ import annotations

import json
import logging
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eden.config import TrainingState, get_device, set_seed
from eden.core.genome import HierarchicalGenome, N_CONN_GENES, N_TYPE_GENES
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.maturity import empirical_stability, maturity_score
from eden.core.network import EDENNetwork, SequenceEDENNetwork, count_parameters
from eden.core.neurogenesis import (
    LocalStagnationDetector,
    apply_neurogenesis_exploration_bump,
    apply_neurogenesis_structural_growth,
)
from eden.metrics import (
    estimate_forward_flops_eden,
    kaplan_meier_style_summary,
    node_activation_redundancy,
)
from eden.optim.eggroll_l1 import differentiation_param_filter, eggroll_l1_step
from eden.optim.eggroll_l2 import eggroll_l2_step

log = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "eden_train_v1"


def _optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_eden_training_checkpoint(
    path: Path,
    model: nn.Module,
    regulator: nn.Module,
    epigenome: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_state: TrainingState,
    loss_hist: list[float],
    epoch_next: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "regulator": regulator.state_dict(),
            "epigenome": epigenome.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_state": {
                "epoch": training_state.epoch,
                "batch": training_state.batch,
                "last_loss": training_state.last_loss,
                "last_accuracy": training_state.last_accuracy,
                "stability_ema": training_state.stability_ema,
                "events": training_state.events[-500:],
            },
            "loss_hist": loss_hist[-5000:],
            "epoch_next": epoch_next,
        },
        path,
    )


def load_eden_training_checkpoint(
    path: Path,
    model: nn.Module,
    regulator: nn.Module,
    epigenome: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, list[float], TrainingState]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    fmt = ckpt.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint format {fmt!r} is not a full training state ({CHECKPOINT_FORMAT!r}). "
            "Use checkpoints saved via --checkpoint-out during training."
        )
    model.load_state_dict(ckpt["model"])
    regulator.load_state_dict(ckpt["regulator"])
    epigenome.load_state_dict(ckpt["epigenome"])
    optimizer.load_state_dict(ckpt["optimizer"])
    model.to(device)
    regulator.to(device)
    epigenome.to(device)
    _optimizer_to_device(optimizer, device)
    ts = ckpt["training_state"]
    st = TrainingState(
        epoch=ts["epoch"],
        batch=ts["batch"],
        last_loss=ts["last_loss"],
        last_accuracy=ts["last_accuracy"],
        stability_ema=ts["stability_ema"],
        events=list(ts.get("events", [])),
    )
    return int(ckpt["epoch_next"]), list(ckpt["loss_hist"]), st


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    epsilon: float = 0.1,
    forward_kw: dict[str, Any] | None = None,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    forward_kw = forward_kw or {}
    model.train()
    out = model(x, **forward_kw)
    if hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    loss = loss_fn(logits, y)
    loss.backward()
    adv = x + epsilon * x.grad.sign()
    return adv.detach()


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def train_eden(
    model: EDENNetwork | SequenceEDENNetwork,
    regulator: GeneRegulator,
    epigenome: HeritableEpigenome,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    dry_run: bool = False,
    l1_every: int = 20,
    l2_every_epochs: int = 5,
    results_dir: Path | None = None,
    stagnation: LocalStagnationDetector | None = None,
    use_eggroll: bool = False,
    log_batches_every: int = 0,
    resume_from: Path | None = None,
    checkpoint_out: Path | None = None,
    checkpoint_every: int = 0,
    maturity_threshold: float = 0.72,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    ``epochs`` is the total number of epochs when starting fresh.
    When ``resume_from`` is set, ``epochs`` is the number of *additional* epochs to run after the checkpoint.
    If ``seed`` is set, it overrides the global ``EDEN_SEED`` for this run only.
    """
    set_seed(seed)
    device = get_device()
    model.to(device)
    regulator.to(device)
    epigenome.to(device)

    opt = torch.optim.Adam(
        list(model.parameters()) + list(regulator.parameters()) + list(epigenome.parameters()),
        lr=lr,
    )
    ce = nn.CrossEntropyLoss()

    if dry_run and resume_from:
        log.warning("dry-run: ignoring --resume")
        resume_from = None

    if resume_from:
        epoch_start, loss_hist, state = load_eden_training_checkpoint(
            resume_from, model, regulator, epigenome, opt, device
        )
        genome = HierarchicalGenome(subgenome_type=model._genome_type, subgenome_conn=model._genome_conn)
        log.info("resumed from %s | next_epoch_index=%d | extra_epochs=%d", resume_from, epoch_start, epochs)
    else:
        gen = torch.Generator()
        gen.manual_seed(0)
        genome = HierarchicalGenome.random(device, generator=gen)
        model.set_genome_buffers(genome)
        loss_hist = []
        state = TrainingState()
        epoch_start = 0

    epoch_end = epoch_start + epochs
    t0 = time.perf_counter()

    if dry_run:
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)
        mask, _ = epigenome()
        reg = regulator()
        m0 = mask[: N_TYPE_GENES + N_CONN_GENES] if mask.shape[0] >= N_TYPE_GENES + N_CONN_GENES else torch.ones(
            N_TYPE_GENES + N_CONN_GENES, device=device
        )
        out = model(xb, reg, m0, state)
        log.info(
            "dry-run OK | logits %s | params %d | device %s",
            list(out.logits.shape),
            count_parameters(model),
            device,
        )
        return {
            "dry_run_ok": True,
            "logits_shape": list(out.logits.shape),
            "params": count_parameters(model),
        }

    val_acc = 0.0
    mat = 0.0
    mvec = torch.ones(N_TYPE_GENES + N_CONN_GENES, device=device)
    red_ema = 0.5
    epochs_to_maturity: int | None = None

    if stagnation is not None:
        stagnation_det: LocalStagnationDetector | None = stagnation
    elif model.flags.neurogenesis:
        stagnation_det = LocalStagnationDetector()
    else:
        stagnation_det = None

    log.info(
        "train start | device=%s | params=%d | epochs %d->%d | batches train=%d val=%d | eggroll=%s | resume=%s",
        device,
        count_parameters(model),
        epoch_start,
        epoch_end,
        len(train_loader),
        len(val_loader),
        use_eggroll,
        resume_from is not None,
    )

    epochs_completed = epoch_start
    for epoch in range(epoch_start, epoch_end):
        state.epoch = epoch
        model.train()
        regulator.train()
        stem_scores = torch.zeros(model.stem_pool.n_stems, device=device)
        grad_accum = 0.0
        n_batches = 0
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            state.batch = batch_idx
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mask, _ = epigenome()
            if mask.shape[0] >= N_TYPE_GENES + N_CONN_GENES:
                mvec = mask[: N_TYPE_GENES + N_CONN_GENES]
            else:
                mvec = torch.ones(N_TYPE_GENES + N_CONN_GENES, device=device)
            reg = regulator()
            out = model(xb, reg, mvec, state)
            loss = ce(out.logits, yb)
            loss.backward()
            gn = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            grad_accum += gn
            n_batches += 1
            opt.step()
            state.last_loss = float(loss.item())
            state.last_accuracy = accuracy_from_logits(out.logits, yb)
            loss_hist.append(state.last_loss)
            stem_scores += state.last_accuracy
            bs = xb.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_correct += int((out.logits.argmax(-1) == yb).sum().item())
            train_total += bs

            if log_batches_every > 0 and (batch_idx + 1) % log_batches_every == 0:
                log.debug(
                    "epoch %d batch %d | loss %.4f | acc %.4f | grad_norm %.4f",
                    epoch + 1,
                    batch_idx + 1,
                    state.last_loss,
                    state.last_accuracy,
                    gn,
                )

            if model.flags.epigenome_drift:
                epigenome.drift_step(True, scale=float(reg["drift_rate_scale"].item()))

            if stagnation_det is not None and stagnation_det.observe(gn):
                state.log_event("neurogenesis_stagnation", grad_norm=gn)
                if model.flags.neurogenesis:
                    new_ps, grow_kind = apply_neurogenesis_structural_growth(model)
                    if new_ps:
                        # Pathway mitosis replaces ``node_proj`` Parameters; stem mitosis only appends a stem.
                        if grow_kind == "neurogenesis_pathway_mitosis":
                            opt = torch.optim.Adam(
                                list(model.parameters())
                                + list(regulator.parameters())
                                + list(epigenome.parameters()),
                                lr=lr,
                            )
                        else:
                            opt.add_param_group({"params": new_ps, "lr": lr})
                        state.log_event(
                            grow_kind or "neurogenesis_growth",
                            n_nodes=model.n_nodes,
                            n_stems=model.stem_pool.n_stems,
                        )
                    else:
                        apply_neurogenesis_exploration_bump(model)
                        state.log_event("neurogenesis_bump", grad_norm=gn)
                    if isinstance(stagnation_det, LocalStagnationDetector):
                        stagnation_det._bad = 0

            if use_eggroll and (batch_idx + 1) % l1_every == 0:
                x_es, y_es = xb.detach(), yb.detach()

                def get_es_loss() -> float:
                    model.eval()
                    with torch.no_grad():
                        mask_es, _ = epigenome()
                        m_es = (
                            mask_es[: N_TYPE_GENES + N_CONN_GENES]
                            if mask_es.shape[0] >= N_TYPE_GENES + N_CONN_GENES
                            else mvec
                        )
                        o = model(x_es, regulator(), m_es, state)
                        return float(ce(o.logits, y_es).item())

                eggroll_l1_step(
                    model, get_es_loss, differentiation_param_filter, lr=0.001, sigma=0.01, n_pairs=20
                )
                model.train()

            if epoch >= 8 and batch_idx == 0:
                model.prune_synapses(0.01)

        stem_scores /= max(n_batches, 1)
        model.stem_pool.update_scores(stem_scores)
        model.stem_pool.apply_retention(epoch)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mask, _ = epigenome()
                mvec = mask[: N_TYPE_GENES + N_CONN_GENES] if mask.shape[0] >= N_TYPE_GENES + N_CONN_GENES else mask
                o = model(xb, regulator(), mvec, state)
                correct += (o.logits.argmax(-1) == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(total, 1)
        with torch.no_grad():
            xb0, _ = next(iter(val_loader))
            xb0 = xb0.to(device)
            mask0, _ = epigenome()
            mv0 = (
                mask0[: N_TYPE_GENES + N_CONN_GENES]
                if mask0.shape[0] >= N_TYPE_GENES + N_CONN_GENES
                else mvec
            )
            o0 = model(xb0, regulator(), mv0, state)
            red_batch = node_activation_redundancy(o0.node_activations)
        red_ema = 0.85 * red_ema + 0.15 * red_batch
        stab = empirical_stability(loss_hist)
        state.stability_ema = 0.9 * state.stability_ema + 0.1 * stab
        mat = maturity_score(val_acc, genome.expressed_ratio(mvec.bool()), state.stability_ema, red_ema)
        if epochs_to_maturity is None and mat >= maturity_threshold:
            epochs_to_maturity = epoch + 1

        t_loss = train_loss_sum / max(train_total, 1)
        t_acc = train_correct / max(train_total, 1)
        n_stem = int(model.stem_pool.active_mask.sum().item())
        log.info(
            "epoch %3d/%d | train loss %.4f acc %.4f | val acc %.4f | maturity %.4f | "
            "expr %.3f | active_stems %d/%d | stability %.3f",
            epoch + 1,
            epoch_end,
            t_loss,
            t_acc,
            val_acc,
            mat,
            genome.expressed_ratio(mvec.bool()),
            n_stem,
            model.stem_pool.n_stems,
            state.stability_ema,
        )

        if use_eggroll and (epoch + 1) % l2_every_epochs == 0:

            def metric() -> float:
                model.eval()
                c2, t2 = 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        mask, _ = epigenome()
                        mv = mask[: N_TYPE_GENES + N_CONN_GENES] if mask.shape[0] >= N_TYPE_GENES + N_CONN_GENES else mvec
                        o = model(xb, regulator(), mv, state)
                        c2 += (o.logits.argmax(-1) == yb).sum().item()
                        t2 += yb.numel()
                return c2 / max(t2, 1)

            eggroll_l2_step(regulator, metric, lr=0.005, sigma=0.05, n_pairs=10, maximize=True)
            model.train()

        epochs_completed = epoch + 1
        if checkpoint_out and checkpoint_every > 0 and epochs_completed % checkpoint_every == 0:
            save_eden_training_checkpoint(
                checkpoint_out,
                model,
                regulator,
                epigenome,
                opt,
                state,
                loss_hist,
                epochs_completed,
            )
            log.info("checkpoint saved | %s | epoch_next=%d", checkpoint_out, epochs_completed)

        if mat > 0.95 and val_acc > 0.9:
            log.info("early stop: maturity %.4f val_acc %.4f", mat, val_acc)
            break

    elapsed = time.perf_counter() - t0
    model.eval()
    final_acc = val_acc
    expr_ratio = float(mvec.float().mean().item()) if mvec.numel() else 1.0

    robust_n = 0
    robust_ok = 0
    for xb, yb in islice(iter(val_loader), 3):
        xb, yb = xb.to(device), yb.to(device)
        mask, _ = epigenome()
        mv = mask[: N_TYPE_GENES + N_CONN_GENES] if mask.shape[0] >= N_TYPE_GENES + N_CONN_GENES else mvec

        for i in range(min(8, xb.size(0))):
            xi = xb[i : i + 1]
            yi = yb[i : i + 1]
            adv = fgsm_attack(
                model,
                xi,
                yi,
                ce,
                0.1,
                forward_kw={"regulator": regulator(), "epigenome_mask": mv, "training_state": state},
            )
            robust_n += 1
            with torch.no_grad():
                oa = model(xi, regulator(), mv, state).logits.argmax(-1)
                ob = model(adv, regulator(), mv, state).logits.argmax(-1)
            if oa == ob:
                robust_ok += 1
    robust = robust_ok / max(robust_n, 1)

    x_flop, _ = next(iter(val_loader))
    x_flop = x_flop[:1].to(device)
    mv_flop = mvec
    flops_est = estimate_forward_flops_eden(model, x_flop, regulator, mv_flop, state)
    surv = kaplan_meier_style_summary(state.events)

    log.info(
        "train done | %.1fs | val_acc=%.4f | epochs=%d | FGSM robust=%.4f | flops_est=%s | report -> %s",
        elapsed,
        final_acc,
        epochs_completed,
        robust,
        flops_est,
        (results_dir / "train_report.json") if results_dir else "—",
    )

    report: dict[str, Any] = {
        "final_val_accuracy": final_acc,
        "epochs_ran": epochs_completed,
        "epochs_to_maturity": epochs_to_maturity,
        "maturity_threshold": maturity_threshold,
        "maturity": mat,
        "activation_redundancy_ema": red_ema,
        "params": count_parameters(model),
        "forward_flops_estimate": flops_est,
        "seconds": elapsed,
        "genome_expr_ratio": expr_ratio,
        "stability": state.stability_ema,
        "adversarial_robustness_fgsm_0.1": robust,
        "survival_event_summary": surv,
        "events_tail": state.events[-20:],
    }
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "train_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if checkpoint_out:
        save_eden_training_checkpoint(
            checkpoint_out,
            model,
            regulator,
            epigenome,
            opt,
            state,
            loss_hist,
            epochs_completed,
        )
        log.info("final checkpoint | %s | epoch_next=%d", checkpoint_out, epochs_completed)

    return report


def save_checkpoint(path: Path, model: nn.Module, regulator: nn.Module, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "regulator": regulator.state_dict(), **extra}, path)


def load_checkpoint(path: Path, model: nn.Module, regulator: nn.Module) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    regulator.load_state_dict(ckpt["regulator"])
    return {k: v for k, v in ckpt.items() if k not in ("model", "regulator")}
