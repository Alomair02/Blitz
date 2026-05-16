"""
Portfolio visualizations for the Blitz GPU reserving project.

The module reads generated Blitz artifacts and produces PNG charts that
highlight GPU workload mapping, throughput, reserve calibration, and portfolio
uncertainty.

Typical usage:
    python reserve_charts.py --outdir docs/charts
    python reserve_charts.py --benchmark --outdir docs/charts
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/blitz-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
N_COMPANIES_DEFAULT = 146
N_BOOT_DEFAULT = 10_000
BLOCK_SIZE_DEFAULT = 256

COLORS = {
    "ink": "#172033",
    "muted": "#697386",
    "grid": "#DDE3EA",
    "naive": "#2F80ED",
    "odp": "#06A77D",
    "target": "#F25F5C",
    "accent": "#F6AE2D",
    "purple": "#4B3F72",
    "teal_dark": "#126782",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FBFCFE",
            "axes.facecolor": "#FBFCFE",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
            "legend.frameon": False,
            "savefig.facecolor": "#FBFCFE",
        }
    )


def read_hip_define(root: Path, name: str, default: int) -> int:
    pattern = re.compile(rf"^\s*#define\s+{re.escape(name)}\s+([0-9]+)\b")
    for source in (root / "ibnr_bootstrap.hip", root / "ibnr_bootstrap_odp.hip"):
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            match = pattern.match(line)
            if match:
                return int(match.group(1))
    return default


def load_constants(root: Path) -> dict[str, int]:
    return {
        "n_companies": read_hip_define(root, "N_COMPANIES", N_COMPANIES_DEFAULT),
        "n_boot": read_hip_define(root, "N_BOOT", N_BOOT_DEFAULT),
        "block_size": read_hip_define(root, "BLOCK_SIZE", BLOCK_SIZE_DEFAULT),
    }


def ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_chart(fig: plt.Figure, outdir: Path, filename: str, dpi: int) -> Path:
    path = outdir / filename
    try:
        fig.tight_layout()
    except RuntimeError:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def shorten(text: str, limit: int = 34) -> str:
    clean = str(text).replace("|", " | ")
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def money_m_formatter(value: float, _pos: int) -> str:
    return f"${value:,.1f}M"


def pct_formatter(value: float, _pos: int) -> str:
    return f"{value:.0f}%"


def require_files(root: Path, filenames: list[str]) -> bool:
    missing = [name for name in filenames if not (root / name).exists()]
    if missing:
        print("Skipping chart; missing generated files: " + ", ".join(missing))
        print("Run: python parse_triangles.py, ./ibnr_bootstrap, ./ibnr_bootstrap_odp, make validate")
        return False
    return True


def plot_gpu_workload_grid(root: Path, outdir: Path, dpi: int) -> Path:
    constants = load_constants(root)
    n_companies = constants["n_companies"]
    n_boot = constants["n_boot"]
    block_size = constants["block_size"]
    blocks_per_company = math.ceil(n_boot / block_size)
    tail_threads = n_boot - block_size * (blocks_per_company - 1)

    active = np.full((n_companies, blocks_per_company), block_size, dtype=float)
    active[:, -1] = tail_threads

    launched_threads = n_companies * blocks_per_company * block_size
    active_threads = n_companies * n_boot
    launch_efficiency = active_threads / launched_threads * 100

    configure_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    cmap = LinearSegmentedColormap.from_list(
        "gpu_blocks", ["#F4D35E", "#06A77D", "#126782"]
    )

    sns.heatmap(
        active,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=block_size,
        cbar_kws={"label": "active threads per GPU block"},
        xticklabels=5,
        yticklabels=False,
    )

    fig.suptitle(
        "GPU Thread Grid: One Reserve Projection per Thread",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.08,
        0.925,
        (
            f"{active_threads:,} active simulation threads per kernel launch | "
            f"{launch_efficiency:.1f}% launch efficiency"
        ),
        ha="left",
        color=COLORS["muted"],
        fontsize=12,
    )
    ax.set_xlabel(f"GPU blocks per company ({blocks_per_company} blocks, {block_size} threads each)")
    ax.set_ylabel(f"Companies ({n_companies} independent loss triangles)")
    ax.text(
        0.995,
        -0.14,
        f"Each row: {n_boot:,} bootstrap trials for one company. Tail block: {tail_threads} active threads.",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["muted"],
        fontsize=12,
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.17)

    return save_chart(fig, outdir, "01_gpu_thread_grid.png", dpi)


def run_command(root: Path, command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def parse_gpu_runtime(output: str) -> tuple[float, float]:
    match = re.search(r"Done in\s+([0-9.]+)s\s+\(([0-9.]+)M projections/s\)", output)
    if not match:
        raise ValueError("Could not parse GPU runtime from executable output.")
    elapsed_seconds = float(match.group(1))
    million_proj_per_second = float(match.group(2))
    return elapsed_seconds, million_proj_per_second


def benchmark_gpu(root: Path, benchmark_csv: Path) -> pd.DataFrame:
    constants = load_constants(root)
    n_companies = constants["n_companies"]
    n_boot = constants["n_boot"]
    block_size = constants["block_size"]
    blocks_per_company = math.ceil(n_boot / block_size)
    active_threads = n_companies * n_boot
    launched_threads = n_companies * blocks_per_company * block_size

    if not (root / "ibnr_bootstrap").exists() or not (root / "ibnr_bootstrap_odp").exists():
        print("Building HIP executables before benchmarking...")
        run_command(root, ["make", "all"])

    engines = [
        ("Naive chain ladder", "./ibnr_bootstrap"),
        ("ODP residual bootstrap", "./ibnr_bootstrap_odp"),
    ]

    rows = []
    for engine_name, executable in engines:
        print(f"Benchmarking {engine_name}...")
        try:
            output = run_command(root, [executable])
        except subprocess.CalledProcessError as exc:
            print(f"Skipping GPU benchmark for {engine_name}; executable failed.")
            if exc.stdout:
                print(exc.stdout.strip())
            if benchmark_csv.exists():
                print(f"Using existing benchmark file: {benchmark_csv}")
                return pd.read_csv(benchmark_csv)
            return pd.DataFrame()
        elapsed, mproj_per_second = parse_gpu_runtime(output)
        rows.append(
            {
                "engine": engine_name,
                "elapsed_seconds": elapsed,
                "million_projections_per_second": mproj_per_second,
                "projections": active_threads,
                "active_threads": active_threads,
                "launched_threads": launched_threads,
                "launch_efficiency_pct": active_threads / launched_threads * 100,
                "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(benchmark_csv, index=False)
    print(f"Wrote {benchmark_csv}")
    return df


def load_or_run_benchmark(root: Path, outdir: Path, run_benchmark: bool) -> pd.DataFrame | None:
    benchmark_csv = outdir / "gpu_benchmark.csv"
    if run_benchmark:
        return benchmark_gpu(root, benchmark_csv)
    if benchmark_csv.exists():
        return pd.read_csv(benchmark_csv)
    print("Skipping throughput chart; run with --benchmark to collect GPU timings.")
    return None


def plot_gpu_throughput(root: Path, outdir: Path, dpi: int, run_benchmark: bool) -> Path | None:
    df = load_or_run_benchmark(root, outdir, run_benchmark)
    if df is None or df.empty:
        return None

    configure_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    palette = [COLORS["naive"], COLORS["odp"]]
    x = np.arange(len(df))
    bars = ax.bar(x, df["million_projections_per_second"], color=palette[: len(df)], width=0.55)

    ax.set_title("GPU Throughput: Millions of Reserve Scenarios per Second", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(df["engine"])
    ax.set_ylabel("Million projections / second")
    ax.set_ylim(0, max(df["million_projections_per_second"].max() * 1.25, 1))

    for bar, (_, row) in zip(bars, df.iterrows()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}M/s\n{row['elapsed_seconds']:.3f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            color=COLORS["ink"],
        )

    projections = int(df["projections"].iloc[0])
    efficiency = float(df["launch_efficiency_pct"].iloc[0])
    ax.text(
        0.02,
        0.96,
        f"{projections:,} reserve projections per run | {efficiency:.1f}% GPU launch efficiency",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color=COLORS["muted"],
    )
    ax.spines[["top", "right"]].set_visible(False)

    return save_chart(fig, outdir, "02_gpu_throughput.png", dpi)


def load_validation_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not require_files(
        root,
        ["ibnr_validation.csv", "ibnr_odp_validation.csv", "ibnr_odp_summary.csv"],
    ):
        raise FileNotFoundError("Missing validation outputs.")

    naive = pd.read_csv(root / "ibnr_validation.csv")
    odp_val = pd.read_csv(root / "ibnr_odp_validation.csv")
    odp_summary = pd.read_csv(root / "ibnr_odp_summary.csv")
    return naive, odp_val, odp_summary


def plot_calibration_lift(root: Path, outdir: Path, dpi: int) -> Path | None:
    try:
        naive, odp_val, odp_summary = load_validation_frames(root)
    except FileNotFoundError:
        return None

    valid = naive["true_ibnr"] > 1
    odp_p50_cover = odp_summary["p50"].to_numpy() >= odp_val["true_ibnr"].to_numpy()

    rows = [
        ("P50", "Naive factor bootstrap", naive.loc[valid, "cover_p50"].mean() * 100, 50),
        ("P75", "Naive factor bootstrap", naive.loc[valid, "cover_p75"].mean() * 100, 75),
        ("P95", "Naive factor bootstrap", naive.loc[valid, "cover_p95"].mean() * 100, 95),
        ("P50", "ODP residual bootstrap", odp_p50_cover[valid.to_numpy()].mean() * 100, 50),
        ("P75", "ODP residual bootstrap", odp_val.loc[valid, "cover_p75"].mean() * 100, 75),
        ("P95", "ODP residual bootstrap", odp_val.loc[valid, "cover_p95"].mean() * 100, 95),
    ]
    df = pd.DataFrame(rows, columns=["percentile", "method", "coverage", "target"])

    configure_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    sns.barplot(
        data=df,
        x="percentile",
        y="coverage",
        hue="method",
        palette=[COLORS["naive"], COLORS["odp"]],
        ax=ax,
    )

    target_df = df.drop_duplicates("percentile")
    ax.scatter(
        target_df["percentile"],
        target_df["target"],
        s=170,
        marker="D",
        color=COLORS["target"],
        label="Target coverage",
        zorder=5,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=10, color=COLORS["ink"])

    ax.set_title("Reserve Calibration Lift from the ODP GPU Bootstrap", pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("Companies covered by reserve percentile")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.04))
    ax.text(
        0.98,
        0.035,
        "Validation set: companies with true IBNR > $1K",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["muted"],
        fontsize=12,
    )
    ax.spines[["top", "right"]].set_visible(False)

    return save_chart(fig, outdir, "03_calibration_lift.png", dpi)


def plot_portfolio_reserves(root: Path, outdir: Path, dpi: int) -> Path | None:
    try:
        naive, odp_val, _odp_summary = load_validation_frames(root)
    except FileNotFoundError:
        return None

    rows = [
        ("True IBNR", naive["true_ibnr"].sum() / 1000),
        ("Naive mean", naive["boot_mean"].sum() / 1000),
        ("ODP best estimate", odp_val["mean_odp"].sum() / 1000),
        ("ODP P75", odp_val["p75_odp"].sum() / 1000),
        ("ODP P95", odp_val["p95_odp"].sum() / 1000),
        ("Posted reserves", naive["posted_reserve"].sum() / 1000),
    ]
    df = pd.DataFrame(rows, columns=["metric", "amount_m"])

    configure_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    palette = [
        COLORS["ink"],
        COLORS["naive"],
        COLORS["odp"],
        COLORS["accent"],
        COLORS["target"],
        COLORS["purple"],
    ]

    sns.barplot(
        data=df,
        x="amount_m",
        y="metric",
        hue="metric",
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
    )

    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + max(df["amount_m"]) * 0.015,
            patch.get_y() + patch.get_height() / 2,
            f"${width:,.1f}M",
            va="center",
            ha="left",
            fontsize=13,
            color=COLORS["ink"],
        )

    ax.set_title("Portfolio Reserve Stack: Model Output vs Realized Outcomes", pad=18)
    ax.set_xlabel("Reserve amount (USD millions)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(money_m_formatter))
    ax.set_xlim(0, max(df["amount_m"]) * 1.18)
    fig.text(
        0.125,
        0.02,
        "Underlying CSV values are stored in thousands of USD.",
        color=COLORS["muted"],
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)

    return save_chart(fig, outdir, "04_portfolio_reserve_stack.png", dpi)


def simple_kde(values: np.ndarray, points: int = 260) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return values, np.zeros_like(values)

    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = values.min(), values.max()
    span = hi - lo
    x = np.linspace(lo - span * 0.1, hi + span * 0.1, points)

    std = values.std(ddof=1)
    bandwidth = 1.06 * std * len(values) ** (-1 / 5)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max(span / 30, 1e-6)

    z = (x[:, None] - values[None, :]) / bandwidth
    y = np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return x, y


def plot_reserve_distributions(root: Path, outdir: Path, dpi: int) -> Path | None:
    constants = load_constants(root)
    n_companies = constants["n_companies"]
    n_boot = constants["n_boot"]
    if not require_files(root, ["ibnr_odp_samples.bin", "ibnr_odp_validation.csv"]):
        return None

    odp_val = pd.read_csv(root / "ibnr_odp_validation.csv")
    samples = np.fromfile(root / "ibnr_odp_samples.bin", dtype=np.float32)
    expected = n_companies * n_boot
    if samples.size != expected:
        print(f"Skipping distribution chart; expected {expected:,} samples, found {samples.size:,}.")
        return None
    samples = samples.reshape(n_companies, n_boot)

    top_idx = odp_val.sort_values("true_ibnr", ascending=False).head(4).index.to_list()

    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    axes = axes.ravel()

    for ax, idx in zip(axes, top_idx):
        row = odp_val.iloc[idx]
        values = samples[idx]
        x, y = simple_kde(values)

        ax.fill_between(x, y, color=COLORS["odp"], alpha=0.22)
        ax.plot(x, y, color=COLORS["odp"], linewidth=2.4)
        ax.axvline(row["true_ibnr"], color=COLORS["ink"], linewidth=2.2, label="True")
        ax.axvline(row["p75_odp"], color=COLORS["accent"], linewidth=2.2, linestyle="--", label="P75")
        ax.axvline(row["p95_odp"], color=COLORS["target"], linewidth=2.2, linestyle="--", label="P95")
        ax.set_title(shorten(row["name"], 34), fontsize=15, pad=10)
        ax.set_xlabel("IBNR reserve ($000s)")
        ax.set_ylabel("density")
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("ODP Bootstrap Reserve Distributions for High-IBNR Companies", y=1.07)
    fig.text(
        0.5,
        0.005,
        "Each curve is built from 10,000 GPU-generated reserve projections.",
        ha="center",
        color=COLORS["muted"],
        fontsize=12,
    )
    fig.tight_layout()

    return save_chart(fig, outdir, "05_odp_reserve_distributions.png", dpi)


def plot_uncertainty_diagnostics(root: Path, outdir: Path, dpi: int) -> Path | None:
    try:
        _naive, odp_val, odp_summary = load_validation_frames(root)
    except FileNotFoundError:
        return None

    df = pd.DataFrame(
        {
            "phi": odp_summary["phi"],
            "cv_pct": odp_summary["cv_pct"],
            "true_ibnr": odp_val["true_ibnr"],
            "cover_p95": odp_val["cover_p95"].map({0: "P95 miss", 1: "P95 covered"}),
            "name": odp_val["name"],
        }
    )
    df["size"] = np.clip(df["true_ibnr"], 1, df["true_ibnr"].quantile(0.95))

    configure_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    sns.scatterplot(
        data=df,
        x="phi",
        y="cv_pct",
        hue="cover_p95",
        size="size",
        sizes=(40, 520),
        alpha=0.78,
        palette={"P95 covered": COLORS["odp"], "P95 miss": COLORS["target"]},
        edgecolor="white",
        linewidth=0.8,
        legend=False,
        ax=ax,
    )

    ax.set_title("ODP Diagnostics: Where Reserve Uncertainty Concentrates", pad=18)
    ax.set_xlabel("Overdispersion phi")
    ax.set_ylabel("Coefficient of variation")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="P95 covered",
            markerfacecolor=COLORS["odp"],
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="P95 miss",
            markerfacecolor=COLORS["target"],
            markersize=9,
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", title="P95 validation")
    ax.text(
        0.98,
        0.06,
        "Marker size scales with true IBNR",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["muted"],
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)

    return save_chart(fig, outdir, "06_uncertainty_diagnostics.png", dpi)


def generate_charts(root: Path, outdir: Path, dpi: int, run_benchmark: bool) -> list[Path]:
    outdir = ensure_outdir(outdir)
    generated: list[Path] = []

    for chart_func in (
        plot_gpu_workload_grid,
        lambda r, o, d: plot_gpu_throughput(r, o, d, run_benchmark),
        plot_calibration_lift,
        plot_portfolio_reserves,
        plot_reserve_distributions,
        plot_uncertainty_diagnostics,
    ):
        path = chart_func(root, outdir, dpi)
        if path is not None:
            generated.append(path)
            print(f"Wrote {path}")

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Blitz reserve-model charts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root containing the generated Blitz artifacts.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "charts",
        help="Directory where PNG charts will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run HIP executables and chart measured GPU throughput.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = generate_charts(
        root=args.root.resolve(),
        outdir=args.outdir.resolve(),
        dpi=args.dpi,
        run_benchmark=args.benchmark,
    )
    if not generated:
        raise SystemExit("No charts were generated.")


if __name__ == "__main__":
    main()
