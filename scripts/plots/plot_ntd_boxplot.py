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
OUTPUT_PATH = PROJECT_ROOT / "manual_inspect_dataset" / "ntd_distribution_boxplot_2.pdf"
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


def load_file_level_ntd_counts() -> list[int]:
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        ntd_counts = [
            int(row["final_not_to_do_count"])
            for row in rows
            if row.get("final_not_to_do_count")
        ]

    if not ntd_counts:
        raise ValueError(f"No file-level statistics found in {DATA_PATH}")
    return ntd_counts


def main() -> None:
    ntd_counts = load_file_level_ntd_counts()

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

    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=180)
    ax.boxplot(
        ntd_counts,
        vert=True,
        patch_artist=True,
        showfliers=False,
        widths=0.42,
        tick_labels=["All files"],
        boxprops={"facecolor": "#4c78a8", "edgecolor": "#2f5f8f", "linewidth": 1.4},
        medianprops={"color": "#f58518", "linewidth": 2.2},
        whiskerprops={"color": "#374151", "linewidth": 1.2},
        capprops={"color": "#374151", "linewidth": 1.2},
    )

    ax.set_ylabel("Number of NTD instructions in file", fontsize=18, labelpad=14)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
