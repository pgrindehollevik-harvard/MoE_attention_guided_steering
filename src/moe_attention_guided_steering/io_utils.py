import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str) -> Path:
    """Create an output directory if it does not exist yet."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(data: Any, path: Path) -> None:
    """Write JSON with stable formatting for easy diffs and review."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
