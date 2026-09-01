"""Command line interface for py-relex."""
from __future__ import annotations

import argparse

from .core import run_relex
from .io import load_input_json, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RELEX Coulomb excitation solver (Python).")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("--output-dir", default=".", help="Directory to write outputs")
    args = parser.parse_args()

    inputs = load_input_json(args.input)
    result = run_relex(inputs)
    paths = save_outputs(result, args.output_dir)

    print("Wrote outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
