"""CLI entry: train, benchmark, ablation, sensitivity, tui."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import torch

from eden.config import AblationFlags, set_seed
from eden.core.epigenome import HeritableEpigenome
from eden.core.genome import GeneRegulator
from eden.core.network import EDENNetwork, SequenceEDENNetwork
from eden.benchmarks import datasets
from eden.benchmarks.runner import run_benchmark_suite
from eden.config_loader import load_train_yaml
from eden.logutil import setup_cli_logging
from eden.training import train_eden


def _flags_from_mechanism(mechanism: str, base: AblationFlags | None = None) -> AblationFlags:
    f = (base or AblationFlags()).copy()
    m = mechanism.lower().replace("-", "_")
    if m == "microglia":
        f.microglia = False
    elif m == "contact_inhibition":
        f.contact_inhibition = False
    elif m == "hox_waves":
        f.hox_waves = False
    elif m == "programmed_apoptosis":
        f.programmed_apoptosis = False
    elif m == "epigenome_drift":
        f.epigenome_drift = False
    elif m == "synaptic_pruning":
        f.synaptic_pruning = False
    elif m == "glia":
        f.glia = False
    elif m == "node_attention":
        f.node_attention = False
    else:
        raise click.BadParameter(f"unknown mechanism: {mechanism}")
    return f


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--dataset", type=str, default="mnist")
@click.option("--epochs", type=int, default=10)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1e-3)
@click.option("--dry-run", is_flag=True)
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--output", type=click.Path(), default="eden_results")
@click.option("--eggroll", is_flag=True, help="Enable L1/L2 Eggroll (slower; spec default)")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logs (per-batch if --log-batch-every set)")
@click.option("-q", "--quiet", is_flag=True, help="Only warnings and errors")
@click.option("--log-file", type=click.Path(), default=None, help="Append logs to this file")
@click.option("--save-log", is_flag=True, help="Also write logs to <output>/train.log")
@click.option(
    "--log-batch-every",
    type=int,
    default=0,
    show_default=True,
    help="Log every N batches at DEBUG (use with -v)",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Full training checkpoint; --epochs is how many *more* epochs to run",
)
@click.option(
    "--checkpoint-out",
    "checkpoint_out",
    type=click.Path(path_type=Path),
    default=None,
    help="Save full state (model, regulator, epigenome, optimizer) after training",
)
@click.option(
    "--checkpoint-every",
    type=int,
    default=0,
    show_default=True,
    help="Also save checkpoint every N epochs (requires --checkpoint-out)",
)
def train(
    dataset: str,
    epochs: int,
    batch_size: int,
    lr: float,
    dry_run: bool,
    config: str | None,
    output: str,
    eggroll: bool,
    verbose: bool,
    quiet: bool,
    log_file: str | None,
    save_log: bool,
    log_batch_every: int,
    resume: Path | None,
    checkpoint_out: Path | None,
    checkpoint_every: int,
) -> None:
    set_seed()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    lf = log_file or (str(out / "train.log") if save_log else None)
    if config:
        ycfg = load_train_yaml(config)
        epochs = int(ycfg.get("epochs", epochs))
        lr = float(ycfg.get("lr", lr))
        batch_size = int(ycfg.get("batch_size", batch_size))
        eggroll = bool(ycfg.get("eggroll", eggroll))
        log_batch_every = int(ycfg.get("log_batch_every", log_batch_every))
        if "verbose" in ycfg:
            verbose = bool(ycfg["verbose"])
    setup_cli_logging(quiet=quiet, verbose=verbose, log_file=lf)
    data_root = Path(os.environ.get("EDEN_DATA", ".data"))
    flags = AblationFlags()
    if dataset == "synth":
        full = datasets.spiral_dataset(600)
        n = len(full)
        tr, va = torch.utils.data.random_split(
            full,
            [int(0.85 * n), n - int(0.85 * n)],
            generator=torch.Generator().manual_seed(0),
        )
        tr_l = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True)
        va_l = torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False)
        model = SequenceEDENNetwork(2, 2, flags=flags)
    elif dataset in ("ecg", "protein"):
        tr_l, va_l, meta = datasets.get_sequence_loaders(dataset, batch_size)
        nc = int(meta["num_classes"])
        sl = int(meta["seq_len"])
        model = SequenceEDENNetwork(nc, sl, flags=flags)
    else:
        tr_l, va_l, meta = datasets.get_torchvision_loaders(dataset, data_root, batch_size)
        model = EDENNetwork(
            int(meta["num_classes"]),
            int(meta["in_channels"]),
            tuple(meta["image_hw"]),
            flags=flags,
        )
    reg = GeneRegulator()
    epi = HeritableEpigenome()
    if checkpoint_every > 0 and checkpoint_out is None:
        raise click.UsageError("--checkpoint-every requires --checkpoint-out")
    rep = train_eden(
        model,
        reg,
        epi,
        tr_l,
        va_l,
        epochs,
        lr=lr,
        dry_run=dry_run,
        results_dir=out,
        use_eggroll=eggroll,
        log_batches_every=log_batch_every,
        resume_from=resume,
        checkpoint_out=checkpoint_out,
        checkpoint_every=checkpoint_every,
    )
    click.echo(json.dumps(rep, indent=2))


@main.command("benchmark")
@click.option("--all", "run_all", is_flag=True)
@click.option("--dataset", multiple=True, type=str)
@click.option("--seeds", type=int, default=1)
@click.option("--epochs", type=int, default=5)
@click.option("--batch-size", type=int, default=64)
@click.option("--output", type=click.Path(), default="eden_results")
@click.option("--quick", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True)
@click.option("--log-file", type=click.Path(), default=None)
@click.option("--save-log", is_flag=True, help="Write logs to <output>/benchmark.log")
@click.option(
    "--log-batch-every",
    type=int,
    default=0,
    help="EDEN: log every N train batches (DEBUG; use -v)",
)
@click.option(
    "--eden-only",
    "eden_only",
    is_flag=True,
    help="Skip MLP/LeNet/ResNet/CNN1D/LSTM baselines (much faster)",
)
def benchmark_cmd(
    run_all: bool,
    dataset: tuple[str, ...],
    seeds: int,
    epochs: int,
    batch_size: int,
    output: str,
    quick: bool,
    verbose: bool,
    quiet: bool,
    log_file: str | None,
    save_log: bool,
    log_batch_every: int,
    eden_only: bool,
) -> None:
    out = Path(output)
    lf = log_file or (str(out / "benchmark.log") if save_log else None)
    setup_cli_logging(quiet=quiet, verbose=verbose, log_file=lf)
    data_root = Path(os.environ.get("EDEN_DATA", ".data"))
    if run_all:
        ds_list = ["mnist", "fashion_mnist", "cifar10", "cifar100", "ecg", "protein"]
    elif dataset:
        ds_list = list(dataset)
    else:
        ds_list = ["mnist"]
    summ = run_benchmark_suite(
        ds_list,
        seeds,
        epochs,
        batch_size,
        data_root,
        out,
        quick=quick,
        log_batches_every=log_batch_every,
        eden_only=eden_only,
    )
    click.echo(json.dumps(summ, indent=2))


@main.command()
@click.option("--mechanism", type=str, required=True)
@click.option("--seeds", type=int, default=3)
@click.option("--epochs", type=int, default=5)
@click.option("--dataset", type=str, default="mnist")
@click.option("--output", type=click.Path(), default="eden_results")
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True)
def ablation(
    mechanism: str, seeds: int, epochs: int, dataset: str, output: str, verbose: bool, quiet: bool
) -> None:
    setup_cli_logging(quiet=quiet, verbose=verbose)
    data_root = Path(os.environ.get("EDEN_DATA", ".data"))
    out = Path(output) / f"ablation_{mechanism}"
    _seq_datasets = {"ecg", "protein"}
    deltas: list[float] = []
    for s in range(seeds):
        set_seed(s)
        flags = _flags_from_mechanism(mechanism)
        if dataset in _seq_datasets:
            tr_l, va_l, meta = datasets.get_sequence_loaders(dataset, 64, seed=s)
            def _make_model(f: AblationFlags) -> SequenceEDENNetwork:
                return SequenceEDENNetwork(int(meta["num_classes"]), int(meta["seq_len"]), flags=f)
        else:
            tr_l, va_l, meta = datasets.get_torchvision_loaders(dataset, data_root, 64)
            def _make_model(f: AblationFlags) -> EDENNetwork:
                return EDENNetwork(int(meta["num_classes"]), int(meta["in_channels"]), tuple(meta["image_hw"]), flags=f)
        reg = GeneRegulator()
        epi = HeritableEpigenome()
        m0 = train_eden(_make_model(AblationFlags()), GeneRegulator(), HeritableEpigenome(), tr_l, va_l, epochs)
        m1 = train_eden(_make_model(flags), reg, epi, tr_l, va_l, epochs)
        deltas.append(m1["final_val_accuracy"] - m0["final_val_accuracy"])
    rep = {"mechanism": mechanism, "delta_accuracy_mean": sum(deltas) / len(deltas), "per_seed": deltas}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ablation.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    click.echo(json.dumps(rep, indent=2))


REG_ATTRS = {
    "tau": "tau",
    "lambda": "lambda_",
    "k": "k_soft",
    "rho": "rho",
    "drift": "drift_rate_scale",
    "apoptosis": "apoptosis_theta",
    "entropy": "entropy_theta",
}


@main.command()
@click.option(
    "--method",
    type=click.Choice(["grid", "morris"]),
    default="grid",
    help="grid: 3-point factor sweep on --param; morris: elementary effects on all 8 regulator scalars",
)
@click.option("--param", "param_name", type=str, default=None)
@click.option("--range", "range_str", type=str, default="0.8,1.2")
@click.option("--epochs", type=int, default=3)
@click.option("--morris-epochs", type=int, default=1, help="Short inner train for each Morris evaluation")
@click.option("--morris-trajectories", type=int, default=4)
@click.option("--dataset", type=str, default="mnist")
@click.option("--output", type=click.Path(), default="eden_results")
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True)
def sensitivity(
    method: str,
    param_name: str | None,
    range_str: str,
    epochs: int,
    morris_epochs: int,
    morris_trajectories: int,
    dataset: str,
    output: str,
    verbose: bool,
    quiet: bool,
) -> None:
    setup_cli_logging(quiet=quiet, verbose=verbose)
    data_root = Path(os.environ.get("EDEN_DATA", ".data"))
    tr_l, va_l, meta = datasets.get_torchvision_loaders(dataset, data_root, 64)

    if method == "morris":
        from eden.sensitivity_morris import morris_elementary_effects, morris_summary_to_jsonable

        out = Path(output) / "sensitivity_morris"
        out.mkdir(parents=True, exist_ok=True)

        run_id = 0

        def run_acc(reg_mod: GeneRegulator) -> float:
            nonlocal run_id
            m = EDENNetwork(
                int(meta["num_classes"]),
                int(meta["in_channels"]),
                tuple(meta["image_hw"]),
                flags=AblationFlags(),
            )
            epi = HeritableEpigenome()
            r = train_eden(
                m,
                reg_mod,
                epi,
                tr_l,
                va_l,
                morris_epochs,
                results_dir=None,
                seed=100_000 + run_id,
            )
            run_id += 1
            return float(r["final_val_accuracy"])

        attrs = list(REG_ATTRS.values())
        mu = morris_elementary_effects(
            run_acc,
            GeneRegulator,
            attrs,
            delta=0.2,
            n_trajectories=morris_trajectories,
        )
        rep = morris_summary_to_jsonable(mu)
        rep["dataset"] = dataset
        rep["morris_epochs"] = morris_epochs
        rep["morris_trajectories"] = morris_trajectories
        with open(out / "sensitivity_morris.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        click.echo(json.dumps(rep, indent=2))
        return

    if not param_name:
        raise click.UsageError("--param is required for --method grid")
    out = Path(output) / f"sensitivity_{param_name}"
    out.mkdir(parents=True, exist_ok=True)
    key = param_name.lower()
    if key not in REG_ATTRS:
        raise click.BadParameter(f"unknown param; try one of {list(REG_ATTRS)}")
    attr = REG_ATTRS[key]
    lo, hi = map(float, range_str.split(","))
    factors = [lo, 1.0, hi]
    results: list[dict[str, float]] = []
    reg0 = GeneRegulator()
    base_tensor = getattr(reg0, attr).detach().clone()
    full_sd = {k: v.clone() for k, v in reg0.state_dict().items()}
    for fac in factors:
        set_seed(0)
        model = EDENNetwork(
            int(meta["num_classes"]),
            int(meta["in_channels"]),
            tuple(meta["image_hw"]),
            flags=AblationFlags(),
        )
        reg = GeneRegulator()
        reg.load_state_dict(full_sd)
        with torch.no_grad():
            getattr(reg, attr).copy_(base_tensor * fac)
        epi = HeritableEpigenome()
        r = train_eden(model, reg, epi, tr_l, va_l, epochs)
        results.append({"factor": fac, "accuracy": r["final_val_accuracy"]})
    var = max(x["accuracy"] for x in results) - min(x["accuracy"] for x in results)
    rep = {"method": "grid", "param": param_name, "variance_span": var, "runs": results}
    with open(out / "sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    click.echo(json.dumps(rep, indent=2))


@main.command("interpret")
@click.option("--dataset", type=str, default="mnist")
@click.option("--output", type=click.Path(), default="eden_results")
@click.option("--genomes", type=int, default=24)
def interpret_cmd(dataset: str, output: str, genomes: int) -> None:
    """Gene-space UMAP vs functional similarity (needs umap-learn + scipy for full correlation)."""
    from eden.config import get_device, set_seed
    from eden.interpret import genome_function_correlation_study

    if dataset not in ("mnist", "fashion_mnist", "cifar10", "cifar100"):
        raise click.BadParameter("use a vision dataset: mnist, fashion_mnist, cifar10, cifar100")
    set_seed(0)
    data_root = Path(os.environ.get("EDEN_DATA", ".data"))
    tr_l, _, meta = datasets.get_torchvision_loaders(dataset, data_root, 64)
    model = EDENNetwork(
        int(meta["num_classes"]),
        int(meta["in_channels"]),
        tuple(meta["image_hw"]),
        flags=AblationFlags(),
    )
    reg = GeneRegulator()
    epi = HeritableEpigenome()
    xb, _ = next(iter(tr_l))
    res = genome_function_correlation_study(model, reg, epi, xb, get_device(), n_genomes=genomes)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "interpret_genome_function.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    click.echo(json.dumps(res, indent=2))
    click.echo(f"wrote {path}")


@main.command()
@click.option("--dataset", type=str, default="mnist")
def tui(dataset: str) -> None:
    try:
        from eden.cli.tui_app import run_tui
    except ImportError as e:
        click.echo(f"TUI unavailable: {e}")
        return
    run_tui(dataset)


if __name__ == "__main__":
    main()
