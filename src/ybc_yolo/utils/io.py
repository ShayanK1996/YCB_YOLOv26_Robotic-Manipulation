"""Path utilities, safe mkdir, hashing."""
from pathlib import Path
import hashlib


def safe_mkdir(path: Path, parents: bool = True) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def path_hash(path: Path, length: int = 16) -> int:
    """Deterministic hash of path string for reproducible splits."""
    h = hashlib.sha256(str(path).encode()).hexdigest()
    return int(h[:length], 16)