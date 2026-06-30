#!/usr/bin/env python3
"""Randomly sample repository files and split them into candidate instructions."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


REPO_PATH_PREFIX = "raw"
DEFAULT_SAMPLE_SIZE = 372
OUTPUT_RAW_COLUMNS = ("repository_owner", "repository_name", "file_url", "content")
OUTPUT_SPLIT_COLUMNS = (
    "repository_owner",
    "repository_name",
    "file_url",
    "sentence_index",
    "sentence_content",
)
SENTENCE_CHUNK_RE = re.compile(r".+?(?:\.(?=\s|$)|$)")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)")
FENCED_CODE_BLOCK_RE = re.compile(r"^\s{0,3}(```|~~~)")
NUMERIC_FRAGMENT_RE = re.compile(r"^[\d\s.,:;+\-/()]+$")
WORD_RE = re.compile(r"[A-Za-z']+")
ENGLISH_HINT_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "build",
    "by",
    "code",
    "commit",
    "config",
    "development",
    "do",
    "file",
    "for",
    "from",
    "guidelines",
    "in",
    "is",
    "it",
    "module",
    "not",
    "on",
    "or",
    "project",
    "pull",
    "repository",
    "run",
    "security",
    "style",
    "test",
    "testing",
    "the",
    "this",
    "to",
    "use",
    "with",
}


@dataclass(frozen=True)
class MetadataRow:
    repository_owner: str
    repository_name: str
    file_url: str


@dataclass(frozen=True)
class CandidateRow:
    source_path: Path
    repository_owner: str
    repository_name: str
    repository_path: str
    file_url: str
    content: str


def resolve_project_root() -> Path:
    script_path = Path(__file__).resolve()
    if len(script_path.parents) >= 3 and script_path.parents[1].name == "scripts":
        return script_path.parents[2]
    return script_path.parent


def configure_csv_field_size_limit() -> None:
    field_size_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_size_limit)
            return
        except OverflowError:
            field_size_limit //= 10


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Sample files from raw CSV dumps and split sampled file contents into "
            "sentence rows."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root containing raw_datasets/, collected_datasets/, and split_dataset/.",
    )
    parser.add_argument(
        "--raw-dir-name",
        default="raw_datasets",
        help="Directory of source CSV files relative to the project root.",
    )
    parser.add_argument(
        "--metadata-dir-name",
        default="collected_datasets",
        help="Directory of optional metadata CSV files relative to the project root.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="split_dataset",
        help="Directory that receives random_files_raw.csv and content_split.csv.",
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        type=Path,
        help=(
            "Raw CSV to sample. May be passed more than once. Defaults to every "
            "*.csv file in raw_datasets/."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        action="append",
        type=Path,
        help=(
            "Metadata CSV used to recover repository file URLs. May be passed more "
            "than once. Defaults to every *.csv file in collected_datasets/."
        ),
    )
    parser.add_argument(
        "--total-sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Total number of files to sample across all input CSVs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--allocation",
        choices=("proportional", "equal"),
        default="proportional",
        help="How to allocate the total sample size across input CSVs.",
    )
    parser.add_argument(
        "--include-non-english",
        action="store_true",
        help="Keep all rows instead of applying the lightweight English text filter.",
    )
    parser.add_argument(
        "--min-sentence-chars",
        type=int,
        default=1,
        help="Drop split sentence rows shorter than this many characters.",
    )
    return parser.parse_args()


def resolve_paths(
    explicit_paths: Sequence[Path] | None,
    default_dir: Path,
) -> list[Path]:
    if explicit_paths:
        paths = [
            path if path.is_absolute() else default_dir.parent / path
            for path in explicit_paths
        ]
    else:
        paths = sorted(default_dir.glob("*.csv"))

    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {joined}")
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {default_dir}")
    return paths


def parse_repo_key(file_path: str) -> tuple[str, str, str]:
    parts = PurePosixPath(file_path).parts
    if len(parts) < 4 or parts[0] != REPO_PATH_PREFIX:
        raise ValueError(f"Unexpected raw file_path format: {file_path}")

    repository_owner = parts[1]
    repository_name = parts[2]
    repository_path = PurePosixPath(*parts[3:]).as_posix()
    if not repository_path:
        raise ValueError(f"Missing repository-relative path in file_path: {file_path}")

    return repository_owner, repository_name, repository_path


def is_likely_english(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 20:
        return False

    ascii_letters = sum(1 for char in letters if char.isascii())
    ascii_ratio = ascii_letters / len(letters)
    if ascii_ratio < 0.85:
        return False

    words = WORD_RE.findall(text.lower())
    if len(words) < 8:
        return False

    english_hint_hits = sum(1 for word in words if word in ENGLISH_HINT_WORDS)
    hint_ratio = english_hint_hits / len(words)
    if english_hint_hits >= 3 and hint_ratio >= 0.02:
        return True
    return english_hint_hits >= 2 and ascii_ratio >= 0.97 and len(words) >= 25


def read_metadata_rows(metadata_paths: Iterable[Path]) -> dict[str, MetadataRow]:
    metadata_by_path: dict[str, MetadataRow] = {}
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                file_url = (row.get("file_url") or "").strip()
                repository_owner = (row.get("repository_owner") or "").strip()
                repository_name = (row.get("repository_name") or "").strip()
                if not file_url or not repository_owner or not repository_name:
                    continue

                metadata = MetadataRow(
                    repository_owner=repository_owner,
                    repository_name=repository_name,
                    file_url=file_url,
                )
                for key_column in ("original_file_path", "file_path"):
                    key = (row.get(key_column) or "").strip()
                    if key:
                        metadata_by_path.setdefault(key, metadata)
    return metadata_by_path


def fallback_file_url(
    repository_owner: str,
    repository_name: str,
    repository_path: str,
) -> str:
    return (
        f"https://github.com/{repository_owner}/{repository_name}/blob/main/"
        f"{repository_path}"
    )


def load_candidates(
    raw_path: Path,
    metadata_by_path: dict[str, MetadataRow],
    include_non_english: bool,
) -> list[CandidateRow]:
    candidates: list[CandidateRow] = []
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"file_path", "content"}
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{raw_path} is missing required columns: {missing_list}")

        for row in reader:
            raw_file_path = (row.get("file_path") or "").strip()
            content = (row.get("content") or "").strip()
            if not raw_file_path or not content:
                continue
            if not include_non_english and not is_likely_english(content):
                continue

            try:
                repository_owner, repository_name, repository_path = parse_repo_key(
                    raw_file_path
                )
            except ValueError:
                continue

            metadata = metadata_by_path.get(raw_file_path)
            if metadata:
                repository_owner = metadata.repository_owner
                repository_name = metadata.repository_name
                file_url = metadata.file_url
            else:
                file_url = fallback_file_url(
                    repository_owner,
                    repository_name,
                    repository_path,
                )

            candidates.append(
                CandidateRow(
                    source_path=raw_path,
                    repository_owner=repository_owner,
                    repository_name=repository_name,
                    repository_path=repository_path,
                    file_url=file_url,
                    content=content,
                )
            )
    return candidates


def largest_remainder_allocation(total: int, weights: dict[Path, int]) -> dict[Path, int]:
    if total < 0:
        raise ValueError("--total-sample-size must be non-negative")
    if total == 0:
        return {path: 0 for path in weights}

    available_total = sum(weights.values())
    if total > available_total:
        raise ValueError(
            f"Cannot sample {total} files; only {available_total} eligible files found."
        )

    quotas = {
        path: total * available / available_total
        for path, available in weights.items()
    }
    allocation = {path: min(math.floor(quota), weights[path]) for path, quota in quotas.items()}
    remaining = total - sum(allocation.values())

    ranked_paths = sorted(
        weights,
        key=lambda path: (quotas[path] - math.floor(quotas[path]), weights[path]),
        reverse=True,
    )
    while remaining:
        advanced = False
        for path in ranked_paths:
            if allocation[path] < weights[path]:
                allocation[path] += 1
                remaining -= 1
                advanced = True
                if remaining == 0:
                    break
        if not advanced:
            raise ValueError("Unable to allocate the requested sample size.")

    return allocation


def equal_allocation(total: int, weights: dict[Path, int]) -> dict[Path, int]:
    if total < 0:
        raise ValueError("--total-sample-size must be non-negative")
    if total > sum(weights.values()):
        raise ValueError(
            f"Cannot sample {total} files; only {sum(weights.values())} eligible files found."
        )

    allocation = {path: 0 for path in weights}
    ordered_paths = sorted(weights)
    while sum(allocation.values()) < total:
        advanced = False
        for path in ordered_paths:
            if allocation[path] < weights[path]:
                allocation[path] += 1
                advanced = True
                if sum(allocation.values()) == total:
                    break
        if not advanced:
            raise ValueError("Unable to allocate the requested sample size.")
    return allocation


def allocate_sample_sizes(
    total: int,
    candidates_by_path: dict[Path, list[CandidateRow]],
    strategy: str,
) -> dict[Path, int]:
    weights = {path: len(candidates) for path, candidates in candidates_by_path.items()}
    if strategy == "equal":
        return equal_allocation(total, weights)
    return largest_remainder_allocation(total, weights)


def sample_candidates(
    candidates_by_path: dict[Path, list[CandidateRow]],
    sample_sizes: dict[Path, int],
    seed: int,
) -> list[CandidateRow]:
    rng = random.Random(seed)
    sampled_rows: list[CandidateRow] = []
    seen_file_urls: set[str] = set()

    for path in sorted(candidates_by_path):
        candidates = list(candidates_by_path[path])
        rng.shuffle(candidates)
        source_rows: list[CandidateRow] = []

        for candidate in candidates:
            if candidate.file_url in seen_file_urls:
                continue
            seen_file_urls.add(candidate.file_url)
            source_rows.append(candidate)
            if len(source_rows) == sample_sizes[path]:
                break

        if len(source_rows) != sample_sizes[path]:
            raise ValueError(
                f"Could only sample {len(source_rows)} rows from {path}; "
                f"needed {sample_sizes[path]}."
            )
        sampled_rows.extend(source_rows)

    rng.shuffle(sampled_rows)
    return sampled_rows


def is_numeric_fragment(text: str) -> bool:
    normalized = text.strip().strip(".").strip()
    normalized = normalized.strip("`*_~[]{}<>")
    normalized = normalized.strip()
    return (
        bool(normalized)
        and any(char.isdigit() for char in normalized)
        and bool(NUMERIC_FRAGMENT_RE.fullmatch(normalized))
    )


def iter_instruction_lines(content: str) -> Iterable[str]:
    in_fenced_code_block = False

    for raw_line in content.splitlines():
        stripped_line = raw_line.strip()
        if FENCED_CODE_BLOCK_RE.match(stripped_line):
            in_fenced_code_block = not in_fenced_code_block
            continue
        if in_fenced_code_block:
            continue
        if not stripped_line:
            continue
        if MARKDOWN_HEADING_RE.match(stripped_line):
            continue

        yield stripped_line


def split_line_into_sentence_units(line: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in SENTENCE_CHUNK_RE.finditer(line)
        if match.group(0).strip()
    ]


def split_into_sentences(content: str, min_sentence_chars: int) -> list[str]:
    """Split one sampled file with the line-granular preprocessing used for inspection."""
    sentences: list[str] = []
    for line in iter_instruction_lines(content):
        for sentence in split_line_into_sentence_units(line):
            if len(sentence) < min_sentence_chars:
                continue
            if is_numeric_fragment(sentence):
                continue
            sentences.append(sentence)
    return sentences


def write_random_files(path: Path, rows: Sequence[CandidateRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_RAW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "repository_owner": row.repository_owner,
                    "repository_name": row.repository_name,
                    "file_url": row.file_url,
                    "content": row.content,
                }
            )


def write_sentence_split(
    path: Path,
    rows: Sequence[CandidateRow],
    min_sentence_chars: int,
) -> int:
    sentence_count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_SPLIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            for sentence_index, sentence in enumerate(
                split_into_sentences(row.content, min_sentence_chars),
                start=1,
            ):
                sentence_count += 1
                writer.writerow(
                    {
                        "repository_owner": row.repository_owner,
                        "repository_name": row.repository_name,
                        "file_url": row.file_url,
                        "sentence_index": sentence_index,
                        "sentence_content": sentence,
                    }
                )
    return sentence_count


def main() -> None:
    configure_csv_field_size_limit()
    args = parse_args()

    project_root = args.project_root.resolve()
    raw_dir = project_root / args.raw_dir_name
    metadata_dir = project_root / args.metadata_dir_name
    output_dir = project_root / args.output_dir_name
    raw_output_path = output_dir / "random_files_raw.csv"
    split_output_path = output_dir / "content_split.csv"

    raw_paths = resolve_paths(args.input_csv, raw_dir)
    metadata_paths = resolve_paths(args.metadata_csv, metadata_dir)
    metadata_by_path = read_metadata_rows(metadata_paths)

    candidates_by_path = {
        raw_path: load_candidates(
            raw_path=raw_path,
            metadata_by_path=metadata_by_path,
            include_non_english=args.include_non_english,
        )
        for raw_path in raw_paths
    }
    empty_sources = [path for path, candidates in candidates_by_path.items() if not candidates]
    if empty_sources:
        joined = ", ".join(str(path) for path in empty_sources)
        raise ValueError(f"No eligible rows found in: {joined}")

    sample_sizes = allocate_sample_sizes(
        total=args.total_sample_size,
        candidates_by_path=candidates_by_path,
        strategy=args.allocation,
    )
    sampled_rows = sample_candidates(candidates_by_path, sample_sizes, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_random_files(raw_output_path, sampled_rows)
    sentence_count = write_sentence_split(
        split_output_path,
        sampled_rows,
        args.min_sentence_chars,
    )

    print(f"Wrote {len(sampled_rows):,} sampled files to {raw_output_path}")
    print(f"Wrote {sentence_count:,} sentence rows to {split_output_path}")


if __name__ == "__main__":
    main()
