from __future__ import annotations

import os
import shutil
from pathlib import Path


def configure_huggingface_cache() -> Path:
    """Select a persistent cache with the most usable free space.

    Explicit HF_HOME always wins. On managed Jupyter servers, /root is commonly
    backed by a small overlay while /home/jovyan is a large persistent mount.
    """
    configured = os.getenv("HF_HOME")
    if configured:
        path = Path(configured).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    default = Path.home() / ".cache" / "huggingface"
    candidates = [default]
    jovyan_home = Path("/home/jovyan")
    if jovyan_home.is_dir():
        candidates.append(jovyan_home / ".cache" / "huggingface")

    usable: list[tuple[int, Path]] = []
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".tempbench-write-test"
            probe.touch(exist_ok=True)
            probe.unlink()
            usable.append((shutil.disk_usage(candidate).free, candidate))
        except OSError:
            continue

    if not usable:
        raise RuntimeError(
            "No writable Hugging Face cache directory found. Set HF_HOME to a "
            "writable persistent path."
        )

    _, selected = max(usable, key=lambda item: item[0])
    os.environ["HF_HOME"] = str(selected)
    return selected
