from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ordered_unique(values):
    return sorted(pd.unique(values))


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    p = df.pivot(index="seq_len", columns="d", values=value)
    p = p.sort_index()
    p = p[sorted(p.columns)]
    return p


def _plot_heatmap(ax, data: pd.DataFrame, title: str, cmap: str = "viridis"):
    im = ax.imshow(data.values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("d")
    ax.set_ylabel("seq_len")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(c) for c in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(i) for i in data.index])
    return im


def _find_latest_csv(result_dir: Path) -> Path:
    csv_files = sorted(result_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No csv files found in: {result_dir}")
    return csv_files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, help="Path to benchmark csv")
    parser.add_argument("--result-dir", type=Path, default=Path("ch4/benchmark_result"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir; default is ch4/benchmark_result/plots/<csv_stem>",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else _find_latest_csv(args.result_dir)
    out_dir = args.out_dir if args.out_dir is not None else args.result_dir / "plots" / csv_path.stem

    df = pd.read_csv(csv_path)
    df = df[df["latency"] > 0].copy()

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Latency vs seq_len (log x), grouped by d
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, base in enumerate(["torch_attention", "triton_attention"]):
        sub = df[df["base"] == base]
        for d in _ordered_unique(sub["d"]):
            ss = sub[sub["d"] == d].sort_values("seq_len")
            axes[i].plot(ss["seq_len"], ss["latency"], marker="o", label=f"d={d}")
        axes[i].set_xscale("log", base=2)
        axes[i].set_title(base)
        axes[i].set_xlabel("seq_len")
        axes[i].grid(alpha=0.3)
    axes[0].set_ylabel("latency (ms)")
    axes[1].legend()
    fig.suptitle("Forward Latency by seq_len")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_lines.png", dpi=180)
    plt.close(fig)

    # 2) Peak memory vs seq_len
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, base in enumerate(["torch_attention", "triton_attention"]):
        sub = df[df["base"] == base]
        for d in _ordered_unique(sub["d"]):
            ss = sub[sub["d"] == d].sort_values("seq_len")
            axes[i].plot(ss["seq_len"], ss["peak_memory"], marker="o", label=f"d={d}")
        axes[i].set_xscale("log", base=2)
        axes[i].set_yscale("log", base=2)
        axes[i].set_title(base)
        axes[i].set_xlabel("seq_len")
        axes[i].grid(alpha=0.3)
    axes[0].set_ylabel("peak_memory (MB)")
    axes[1].legend()
    fig.suptitle("Peak Memory by seq_len")
    fig.tight_layout()
    fig.savefig(out_dir / "memory_lines.png", dpi=180)
    plt.close(fig)

    # 3) Speedup heatmap (torch / triton)
    t = _pivot(df[df["base"] == "torch_attention"], "latency")
    r = _pivot(df[df["base"] == "triton_attention"], "latency")
    common_idx = t.index.intersection(r.index)
    common_cols = t.columns.intersection(r.columns)
    speedup = t.loc[common_idx, common_cols] / r.loc[common_idx, common_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = _plot_heatmap(ax, speedup, "Speedup Heatmap (torch/triton)", cmap="magma")
    for y in range(speedup.shape[0]):
        for x in range(speedup.shape[1]):
            val = speedup.values[y, x]
            ax.text(x, y, f"{val:.1f}x", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_heatmap.png", dpi=180)
    plt.close(fig)

    # 4) Memory ratio heatmap (torch/triton)
    tm = _pivot(df[df["base"] == "torch_attention"], "peak_memory")
    rm = _pivot(df[df["base"] == "triton_attention"], "peak_memory")
    common_idx = tm.index.intersection(rm.index)
    common_cols = tm.columns.intersection(rm.columns)
    mem_ratio = tm.loc[common_idx, common_cols] / rm.loc[common_idx, common_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = _plot_heatmap(ax, mem_ratio, "Memory Ratio Heatmap (torch/triton)", cmap="cividis")
    for y in range(mem_ratio.shape[0]):
        for x in range(mem_ratio.shape[1]):
            val = mem_ratio.values[y, x]
            ax.text(x, y, f"{val:.1f}x", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label="memory ratio")
    fig.tight_layout()
    fig.savefig(out_dir / "memory_ratio_heatmap.png", dpi=180)
    plt.close(fig)

    # Text summary
    s_flat = speedup.stack().sort_values(ascending=False)
    m_flat = mem_ratio.stack().sort_values(ascending=False)

    summary_lines = []
    summary_lines.append(f"rows_analyzed={len(df)}")
    summary_lines.append(f"speedup_mean={speedup.values.mean():.3f}x")
    summary_lines.append(f"speedup_median={pd.Series(speedup.values.ravel()).median():.3f}x")
    summary_lines.append(f"speedup_min={speedup.values.min():.3f}x")
    summary_lines.append(f"speedup_max={speedup.values.max():.3f}x")
    summary_lines.append(f"memory_ratio_mean={mem_ratio.values.mean():.3f}x")
    summary_lines.append(f"memory_ratio_min={mem_ratio.values.min():.3f}x")
    summary_lines.append(f"memory_ratio_max={mem_ratio.values.max():.3f}x")

    top_speed = s_flat.head(5)
    summary_lines.append("top_speedup_cases=")
    for (seq_len, d), val in top_speed.items():
        summary_lines.append(f"  seq_len={seq_len}, d={d}, speedup={val:.3f}x")

    top_mem = m_flat.head(5)
    summary_lines.append("top_memory_ratio_cases=")
    for (seq_len, d), val in top_mem.items():
        summary_lines.append(f"  seq_len={seq_len}, d={d}, ratio={val:.3f}x")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"CSV used: {csv_path}")
    print(f"Saved plots and summary to: {out_dir}")


if __name__ == "__main__":
    main()
