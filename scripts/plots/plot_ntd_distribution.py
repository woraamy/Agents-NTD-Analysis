import csv
import os
from pathlib import Path


def resolve_project_root() -> Path:
    script_path = Path(__file__).resolve()
    if len(script_path.parents) >= 3 and script_path.parents[1].name == "scripts":
        return script_path.parents[2]
    return script_path.parent


PROJECT_ROOT = resolve_project_root()
DATA_PATH = PROJECT_ROOT / "manual_inspect_dataset" / "ntd_file_level_stats_2.csv"
OUTPUT_PATH = PROJECT_ROOT / "manual_inspect_dataset" / "ntd_distribution_matplotlib_2.pdf"
MPL_CONFIG_PATH = PROJECT_ROOT / ".matplotlib"
XDG_CACHE_PATH = PROJECT_ROOT / ".cache"

MPL_CONFIG_PATH.mkdir(exist_ok=True)
XDG_CACHE_PATH.mkdir(exist_ok=True)
(XDG_CACHE_PATH / "fontconfig").mkdir(exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_PATH))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_PATH))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_file_level_ntd_percentages() -> list[float]:
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        ntd_pct = [
            float(row["final_not_to_do_pct_of_file"])
            for row in rows
            if row.get("final_not_to_do_pct_of_file")
        ]

    if not ntd_pct:
        raise ValueError(f"No file-level statistics found in {DATA_PATH}")
    return ntd_pct


def main() -> None:
    ntd_pct = load_file_level_ntd_percentages()

    bins = [
        ("0%", lambda value: value == 0),
        (">0-5%", lambda value: 0 < value <= 5),
        (">5-10%", lambda value: 5 < value <= 10),
        (">10-15%", lambda value: 10 < value <= 15),
        (">15-20%", lambda value: 15 < value <= 20),
        (">20-30%", lambda value: 20 < value <= 30),
        (">30-50%", lambda value: 30 < value <= 50),
        (">50%", lambda value: value > 50),
    ]

    labels = [label for label, _ in bins]
    counts = [sum(1 for value in ntd_pct if predicate(value)) for _, predicate in bins]
    percents = [count / len(ntd_pct) * 100 for count in counts]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#6b7280",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    bar_color = "#4c78a8"
    bars = ax.barh(labels, counts, color=bar_color, edgecolor="#2f5f8f", linewidth=0.8)

    ax.set_xlabel("Number of files", fontsize=18, labelpad=14)
    ax.set_ylabel("NTD instructions as % of all instructions in a file", fontsize=18, labelpad=14)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(0, max(counts) * 1.24 if max(counts) else 1)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    for bar, count, percent in zip(bars, counts, percents):
        ax.text(
            bar.get_width() + max(max(counts) * 0.015, 0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({percent:.2f}%)",
            ha="left",
            va="center",
            fontsize=11,
            color="#111827",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
