"""Validate OpenSemCom dataset manifests."""

from __future__ import annotations

import argparse
import json

from opensemcom.manifest import validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an OpenSemCom dataset manifest.")
    parser.add_argument("manifest")
    parser.add_argument("--allow-non-scratch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_manifest(args.manifest, require_scratch=not args.allow_non_scratch)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
