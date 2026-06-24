"""Build OpenSemCom manifests from real scratch datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opensemcom.manifest import build_real_manifest, default_roots, validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a real-data OpenSemCom manifest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", action="append", default=[], help="Scratch dataset root. May be repeated.")
    parser.add_argument("--project-root", default="/home/nickyun/links/scratch/new_study/opensemcom")
    parser.add_argument("--max-per-source", type=int, default=512)
    parser.add_argument("--max-calibration-per-class", type=int, default=64)
    parser.add_argument("--max-eval-per-class", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [Path(root) for root in args.root] if args.root else default_roots(args.project_root)
    result = build_real_manifest(
        output=args.output,
        roots=roots,
        max_per_source=args.max_per_source,
        max_calibration_per_class=args.max_calibration_per_class,
        max_eval_per_class=args.max_eval_per_class,
    )
    summary = validate_manifest(args.output)
    print(
        json.dumps(
            {
                "manifest": str(Path(args.output).expanduser().resolve()),
                "summary": summary,
                "availability": result.availability,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
