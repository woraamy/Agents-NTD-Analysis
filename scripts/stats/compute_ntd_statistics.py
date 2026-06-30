#!/usr/bin/env python3
"""Compute descriptive statistics for inspected not-to-do instructions."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FILE_LEVEL_COLUMNS = (
    "repository_owner",
    "repository_name",
    "file_url",
    "candidate_instruction_count",
    "final_not_to_do_count",
    "final_not_to_do_pct_of_file",
    "has_final_not_to_do",
    "first_sentence_index",
    "last_sentence_index",
)
SUMMARY_COLUMNS = ("statistic", "value")


@dataclass
class FileStats:
    repository_owner: str
    repository_name: str
    file_url: str
    candidate_instruction_count: int = 0
    final_not_to_do_count: int = 0
    first_sentence_index: int | None = None
    last_sentence_index: int | None = None

    @property
    def final_not_to_do_pct_of_file(self) -> float:
        if self.candidate_instruction_count == 0:
            return 0.0
        return self.final_not_to_do_count / self.candidate_instruction_count * 100


def resolve_project_root() -> Path:
    script_path = Path(__file__).resolve()
    if len(script_path.parents) >= 3 and script_path.parents[1].name == "scripts":
        return script_path.parents[2]
    return script_path.parent


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    parser = argparse.ArgumentParser(
        description="Compute file-level and overall NTD statistics from manual inspection."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root containing manual_inspect_dataset/ and split_dataset/.",
    )
    parser.add_argument(
        "--inspection-csv",
        type=Path,
        default=project_root / "manual_inspect_dataset" / "ntd_final_inspection.csv",
        help="Manual inspection CSV with a Final column.",
    )
    parser.add_argument(
        "--content-split-csv",
        type=Path,
        default=project_root / "split_dataset" / "content_split.csv",
        help="Optional sentence split CSV used for coverage statistics.",
    )
    parser.add_argument(
        "--file-level-output",
        type=Path,
        default=project_root / "manual_inspect_dataset" / "ntd_file_level_stats_2.csv",
        help="Output CSV for one row per inspected file.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_root / "manual_inspect_dataset" / "ntd_summary_stats.csv",
        help="Output CSV for overall descriptive statistics.",
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def parse_final_label(value: str | None) -> bool:
    text = (value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y", "ntd", "not_to_do", "not-to-do"}


def parse_sentence_index(value: str | None) -> int | None:
    try:
        return int((value or "").strip())
    except ValueError:
        return None


def load_file_stats(inspection_csv: Path) -> tuple[list[FileStats], int, int]:
    stats_by_file: OrderedDict[str, FileStats] = OrderedDict()
    inspected_rows = 0
    final_not_to_do_rows = 0

    with inspection_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"repository_owner", "repository_name", "file_url", "Final"}
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{inspection_csv} is missing required columns: {missing_list}")

        for fallback_index, row in enumerate(reader, start=1):
            inspected_rows += 1
            file_url = (row.get("file_url") or "").strip() or f"missing-file-url-{fallback_index}"
            file_stats = stats_by_file.get(file_url)
            if file_stats is None:
                file_stats = FileStats(
                    repository_owner=(row.get("repository_owner") or "").strip(),
                    repository_name=(row.get("repository_name") or "").strip(),
                    file_url=file_url,
                )
                stats_by_file[file_url] = file_stats

            file_stats.candidate_instruction_count += 1
            sentence_index = parse_sentence_index(row.get("sentence_index"))
            if sentence_index is not None:
                if file_stats.first_sentence_index is None:
                    file_stats.first_sentence_index = sentence_index
                file_stats.last_sentence_index = sentence_index

            if parse_final_label(row.get("Final")):
                final_not_to_do_rows += 1
                file_stats.final_not_to_do_count += 1

    return list(stats_by_file.values()), inspected_rows, final_not_to_do_rows


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100


def format_number(value: int | float | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def median(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def build_summary_rows(
    file_stats_rows: Sequence[FileStats],
    inspected_rows: int,
    final_not_to_do_rows: int,
    content_split_rows: int | None,
) -> list[tuple[str, int | float | str]]:
    file_count = len(file_stats_rows)
    files_with_ntd = sum(1 for row in file_stats_rows if row.final_not_to_do_count > 0)
    ntd_counts = [row.final_not_to_do_count for row in file_stats_rows]
    ntd_percentages = [row.final_not_to_do_pct_of_file for row in file_stats_rows]
    candidate_counts = [row.candidate_instruction_count for row in file_stats_rows]

    rows: list[tuple[str, int | float | str]] = [
        ("inspected_candidate_instruction_count", inspected_rows),
        ("inspected_file_count", file_count),
        ("final_not_to_do_count", final_not_to_do_rows),
        (
            "final_not_to_do_pct_of_inspected_candidates",
            percent(final_not_to_do_rows, inspected_rows),
        ),
        ("files_with_final_not_to_do_count", files_with_ntd),
        ("files_with_final_not_to_do_pct", percent(files_with_ntd, file_count)),
        ("mean_candidate_instruction_count_per_file", mean(candidate_counts)),
        ("median_candidate_instruction_count_per_file", median(candidate_counts)),
        ("mean_final_not_to_do_count_per_file", mean(ntd_counts)),
        ("median_final_not_to_do_count_per_file", median(ntd_counts)),
        ("max_final_not_to_do_count_per_file", max(ntd_counts, default=0)),
        ("mean_final_not_to_do_pct_of_file", mean(ntd_percentages)),
        ("median_final_not_to_do_pct_of_file", median(ntd_percentages)),
        ("max_final_not_to_do_pct_of_file", max(ntd_percentages, default=0.0)),
    ]

    if content_split_rows is not None:
        rows.extend(
            [
                ("content_split_candidate_instruction_count", content_split_rows),
                (
                    "inspection_coverage_pct_of_content_split",
                    percent(inspected_rows, content_split_rows),
                ),
                (
                    "content_split_minus_inspected_count",
                    content_split_rows - inspected_rows,
                ),
            ]
        )

    return rows


def write_file_level_stats(path: Path, rows: Sequence[FileStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILE_LEVEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "repository_owner": row.repository_owner,
                    "repository_name": row.repository_name,
                    "file_url": row.file_url,
                    "candidate_instruction_count": row.candidate_instruction_count,
                    "final_not_to_do_count": row.final_not_to_do_count,
                    "final_not_to_do_pct_of_file": f"{row.final_not_to_do_pct_of_file:.6f}",
                    "has_final_not_to_do": int(row.final_not_to_do_count > 0),
                    "first_sentence_index": row.first_sentence_index or "",
                    "last_sentence_index": row.last_sentence_index or "",
                }
            )


def write_summary(path: Path, rows: Sequence[tuple[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for statistic, value in rows:
            writer.writerow({"statistic": statistic, "value": format_number(value)})


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    inspection_csv = resolve_path(args.inspection_csv, project_root)
    content_split_csv = resolve_path(args.content_split_csv, project_root)
    file_level_output = resolve_path(args.file_level_output, project_root)
    summary_output = resolve_path(args.summary_output, project_root)

    file_stats_rows, inspected_rows, final_not_to_do_rows = load_file_stats(inspection_csv)
    content_split_rows = count_csv_rows(content_split_csv)
    summary_rows = build_summary_rows(
        file_stats_rows=file_stats_rows,
        inspected_rows=inspected_rows,
        final_not_to_do_rows=final_not_to_do_rows,
        content_split_rows=content_split_rows,
    )

    write_file_level_stats(file_level_output, file_stats_rows)
    write_summary(summary_output, summary_rows)

    print(f"Wrote {len(file_stats_rows):,} file-level rows to {file_level_output}")
    print(f"Wrote summary statistics to {summary_output}")


if __name__ == "__main__":
    main()
