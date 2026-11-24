"""CSV (always) + TensorBoard (if installed) scalar logging."""

from __future__ import annotations

import csv
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str | Path | None, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir) if log_dir else None
        self.rows: list[dict] = []
        self.keys: list[str] = []
        self.tb = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if use_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter

                    self.tb = SummaryWriter(str(self.log_dir))
                except Exception:  # tensorboard not installed: CSV only
                    self.tb = None

    def log(self, step: int, scalars: dict[str, float]) -> None:
        row = {"iteration": step, **{k: float(v) for k, v in scalars.items()}}
        self.rows.append(row)
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)
        if not self.log_dir:
            return
        path = self.log_dir / "progress.csv"
        new_keys = [k for k in row if k not in self.keys]
        if new_keys or not path.exists():
            self.keys += new_keys
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.keys, restval="")
                w.writeheader()
                w.writerows(self.rows)
        else:
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.keys, restval="").writerow(row)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
