"""Write rendered frames to mp4 (imageio + ffmpeg), falling back to GIF or raw npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_video(frames, path: str | Path, fps: int = 50) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        npz = path.with_suffix(".npz")
        np.savez_compressed(npz, frames=np.asarray(frames))
        return npz
    try:
        imageio.mimwrite(str(path), list(frames), fps=fps, macro_block_size=1)
        return path
    except Exception:  # no ffmpeg backend (pip install imageio-ffmpeg) -> GIF
        gif = path.with_suffix(".gif")
        imageio.mimwrite(str(gif), list(frames)[::2], duration=2.0 / fps)
        return gif
