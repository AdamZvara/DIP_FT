"""
Generate a self-contained HTML viewer for a training_data.jsonl file.

Author: Adam Zvara (xzvara01)
Date: 04/2026
"""

import json
import os
import tempfile
from pathlib import Path

import mlflow

from mlflow_export.dataset_html import _build_html


def log_training_data_html(run_dir: Path) -> None:
    """Build a self-contained HTML viewer for training_data.jsonl and log it as an artifact."""
    path = run_dir / "training_data.jsonl"
    if not path.exists():
        return

    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        return

    html = _build_html(records)

    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir) / "training_data.html"
    tmp_path.write_text(html, encoding="utf-8")

    mlflow.log_artifact(str(tmp_path), artifact_path="html")
    tmp_path.unlink()
    os.rmdir(tmpdir)
