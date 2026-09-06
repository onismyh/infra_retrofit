from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import ROOT

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coal_retrofit.reporting import summarize_results_directory, write_results_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a results directory as markdown.")
    parser.add_argument("results_dir", type=Path, help="Directory containing experiment outputs")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional markdown output file. If omitted, the summary is printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        markdown = summarize_results_directory(args.results_dir)
        sys.stdout.write(markdown)
        return
    output_path = write_results_markdown(args.results_dir, args.output)
    sys.stdout.write(f"Wrote markdown summary to {output_path}\n")


if __name__ == "__main__":
    main()
