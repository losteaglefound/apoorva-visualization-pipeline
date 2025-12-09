"""Helper utility functions"""

import json
from typing import Any, Dict, List
from pathlib import Path


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file"""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def normalize_path(path: str, base_dir: Path) -> Path:
    """Normalize a path relative to base directory"""
    if path.startswith("/"):
        return Path(path)
    return base_dir / path


def ensure_directory(path: Path) -> None:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

