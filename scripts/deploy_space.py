#!/usr/bin/env python3
"""Upload the app from a local checkout to a Hugging Face Space.

The Space is not linked to this repo; this script copies the files the app needs into the Space's own git repo.

    huggingface-cli login
    python scripts/deploy_space.py <user>/<space> [<checkout dir>] [--message "..."]

Then add OPENROUTER_API_KEY as a secret in the Space settings. Hugging Face now requires a PRO subscription to create
a new Gradio Space on free CPU hardware; an existing Gradio Space can be reused instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ALLOW_PATTERNS = [
    "README.md",
    "app.py",
    "app/*.py",
    "app/**/*.py",
    "app/science_skills/**",
    "requirements.txt",
    "config.yaml",
    "LICENSE",
    "NOTICE",
]
IGNORE_PATTERNS = ["**/__pycache__/**", "*.pyc"]


def deploy_space(space_id: str, source: str | Path = ".", message: str = "Deploy co-scientist", client=None) -> str:
    """Create the Space if needed, upload the app files and return the Space URL."""
    if client is None:
        from huggingface_hub import HfApi

        client = HfApi()
    client.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    client.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(source),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=message,
    )
    return f"https://huggingface.co/spaces/{space_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("space_id", help="Space to deploy to, as <user>/<space>")
    parser.add_argument("source", nargs="?", default=".", help="checkout to upload from (default: current directory)")
    parser.add_argument("--message", default="Deploy co-scientist", help="commit message in the Space repo")
    args = parser.parse_args()
    print(deploy_space(args.space_id, args.source, args.message))
    print("Add OPENROUTER_API_KEY as a secret in the Space settings if it is not set.")


if __name__ == "__main__":
    main()
