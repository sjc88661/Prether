#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Download the two backbone models used by Prether."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer


PROTT5_REPO = "Rostlab/prot_t5_xl_uniref50"
PROPRIME_REPO = "AI4Protein/ProPrime_650M_OGT_Prediction"


def download_prott5(target_dir: Path) -> None:
    print(f"[INFO] Downloading {PROTT5_REPO} -> {target_dir}")
    tokenizer = T5Tokenizer.from_pretrained(PROTT5_REPO, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_REPO)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


def download_proprime(target_dir: Path) -> None:
    print(f"[INFO] Downloading {PROPRIME_REPO} -> {target_dir}")
    tokenizer = AutoTokenizer.from_pretrained(PROPRIME_REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(PROPRIME_REPO, trust_remote_code=True)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Download Prether backbone models.")
    ap.add_argument(
        "--repo_root",
        type=Path,
        default=repo_root,
        help="Prether repository root. Defaults to the parent of this script.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    prott5_dir = repo_root / "prot_t5_xl_uniref50"
    proprime_dir = repo_root / "ProPrime_650M_OGT_Prediction"

    prott5_dir.mkdir(parents=True, exist_ok=True)
    proprime_dir.mkdir(parents=True, exist_ok=True)

    download_prott5(prott5_dir)
    download_proprime(proprime_dir)

    print("[INFO] Backbone download complete.")
    print(f"[INFO] ProtT5 saved at: {prott5_dir}")
    print(f"[INFO] ProPrime saved at: {proprime_dir}")


if __name__ == "__main__":
    main()
