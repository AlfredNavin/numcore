#!/usr/bin/env python3
"""P1 charts: turn the final benchmark JSON into charts + a summary table.

    python3 scripts/plot_p1.py results        (run_final_bench.sh calls this)

Reads (from the given folder, written by scripts/run_final_bench.sh):
    p1_release.json, p1_native_1thread.json, p1_native_threads.json
Writes:
    p1_gflops.png   GFLOPS of all 6 versions at 1024 and 2048 (the P1 chart)
    p1_threads.png  GFLOPS vs thread count, version 5 vs Eigen
    p1_summary.md   the same numbers as a table, with speedups (for notes/write-up)

Every number is the MEDIAN of the repetitions (less affected by one slow run
than the mean).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

# Colours: one per matrix size (validated categorical pair, colourblind-safe).
SIZE_COLOURS = {1024: "#2a78d6", 2048: "#eb6834"}
# Text and grid in neutral inks, never the series colours.
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIZES = (1024, 2048)
THREADS_MAX = 20

# (label, json file, benchmark name pattern). {n} is filled with the size.
VERSIONS = [
    ("1  Naive (i-j-k)",              "p1_release",        "BM_Naive/{n}"),
    ("2  Loop order (i-k-j)",         "p1_release",        "BM_Ikj/{n}"),
    ("3  Tiled (128)",                "p1_release",        "BM_Tiled/n:{n}/tile:128"),
    ("4  Tiled + -march=native",      "p1_native_1thread", "BM_Tiled/n:{n}/tile:128"),
    ("5  Multithreaded (20 threads)", "p1_native_threads", "BM_Parallel/n:{n}/threads:20/real_time"),
    ("6  Eigen (1 thread)",           "p1_native_1thread", "BM_Eigen/n:{n}/threads:1/real_time"),
    ("6  Eigen (20 threads)",         "p1_native_threads", "BM_Eigen/n:{n}/threads:20/real_time"),
]


def load_medians(path: Path) -> dict[str, float]:
    """benchmark name -> median GFLOPS.

    With repetitions >= 2 Google Benchmark adds aggregate rows (mean, median,
    stddev, cv) and we take the median. With REPS=1 there are no aggregates,
    so the single run is used instead.
    """
    data = json.loads(path.read_text())
    medians, single = {}, {}
    for b in data["benchmarks"]:
        if b.get("aggregate_name") == "median":
            medians[b["run_name"]] = b["FLOPS"] / 1e9
        elif b.get("run_type") == "iteration":
            single[b["run_name"]] = b["FLOPS"] / 1e9
    return medians or single


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_versions(results, out: Path):
    labels = [v[0] for v in VERSIONS]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)

    bar_h = 0.38
    for s, n in enumerate(SIZES):
        ys = [i + (s - 0.5) * (bar_h + 0.02) for i in range(len(VERSIONS))]
        vals = [results[(label, n)] for label in labels]
        ax.barh(ys, vals, height=bar_h, color=SIZE_COLOURS[n], label=f"{n} x {n}")
        for y, v in zip(ys, vals):
            ax.text(v + 1, y, f"{v:.1f}", va="center", fontsize=9, color=TEXT_2)

    ax.set_yticks(range(len(VERSIONS)), labels, color=TEXT, fontsize=10)
    ax.invert_yaxis()  # version 1 at the top
    ax.set_xlabel("GFLOPS (higher is better, median of repetitions)", color=TEXT_2)
    ax.set_title("Matrix multiply, double precision, i7-12700H",
                 color=TEXT, loc="left", fontsize=13, pad=12)
    # Legend in the title row, so it never covers a bar or its label.
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2,
              labelcolor=TEXT)
    ax.set_xlim(0, max(results.values()) * 1.12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_threads(threads_json, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE, sharey=True)
    series = [("numcore v5 (tiled + OpenMP)", "BM_Parallel", "#2a78d6", "o"),
              ("Eigen", "BM_Eigen", "#eb6834", "s")]
    for ax, n in zip(axes, SIZES):
        style_axes(ax)
        for name, prefix, colour, marker in series:
            pts = sorted(
                (int(k.split("threads:")[1].split("/")[0]), v)
                for k, v in threads_json.items()
                if k.startswith(f"{prefix}/n:{n}/")
            )
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colour, linewidth=2, marker=marker, markersize=7,
                    label=name)
        # P-core / hyperthread / E-core boundaries on this CPU.
        for x, txt in ((6, "6 P-cores"), (12, "+ hyperthreads"), (20, "+ 8 E-cores")):
            ax.axvline(x, color=GRID, linewidth=1, zorder=0)
            ax.text(x, 1.0, txt, transform=ax.get_xaxis_transform(), ha="right",
                    va="bottom", fontsize=8, color=TEXT_2)
        ax.set_title(f"{n} x {n}", color=TEXT, loc="left", fontsize=11, pad=16)
        ax.set_xlabel("threads", color=TEXT_2)
        ax.set_xticks([1, 2, 4, 6, 8, 12, 20])
    top = max(threads_json.values()) * 1.12  # headroom so no point is clipped
    for ax in axes:
        ax.set_ylim(0, top)
    axes[0].set_ylabel("GFLOPS", color=TEXT_2)
    # One legend for both panels, in the title row, clear of the lines.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncols=2,
               labelcolor=TEXT)
    fig.suptitle("Scaling with threads (wall-clock time)", color=TEXT, x=0.01,
                 ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_summary(results, out: Path):
    lines = ["| Version | 1024 GFLOPS | 2048 GFLOPS | 2048 speedup vs naive | 2048 vs previous row |",
             "|---|---|---|---|---|"]
    naive = results[(VERSIONS[0][0], 2048)]
    prev = None
    for label, _, _ in VERSIONS:
        a, b = results[(label, 1024)], results[(label, 2048)]
        vs_prev = f"{b / prev:.2f}x" if prev else "-"
        lines.append(f"| {label} | {a:.1f} | {b:.1f} | {b / naive:.1f}x | {vs_prev} |")
        prev = b
    out.write_text("\n".join(lines) + "\n")


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    files = {name: load_medians(folder / f"{name}.json")
             for name in ("p1_release", "p1_native_1thread", "p1_native_threads")}

    results = {}
    for label, file, pattern in VERSIONS:
        for n in SIZES:
            key = pattern.format(n=n)
            if key not in files[file]:
                sys.exit(f"missing {key} in {file}.json: did the benchmark run finish?")
            results[(label, n)] = files[file][key]

    plot_versions(results, folder / "p1_gflops.png")
    plot_threads(files["p1_native_threads"], folder / "p1_threads.png")
    write_summary(results, folder / "p1_summary.md")
    print((folder / "p1_summary.md").read_text())
    print(f"Charts: {folder / 'p1_gflops.png'}, {folder / 'p1_threads.png'}")


if __name__ == "__main__":
    main()
