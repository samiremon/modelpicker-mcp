from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    dataset_dir: Path
    cache_dir: Path
    cpu_threads: int = 2
    max_rows: int = 1000
    max_import_rows: int = 10000
    max_models: int = 5
    max_file_bytes: int = 20 * 1024 * 1024
    max_text_chars: int = 20000
    max_model_download_mb: int = 1500
    min_available_ram_mb: int = 1024
    max_worker_ram_mb: int = 4096
    max_job_seconds: int = 3600

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.getenv("MODELPICKER_HOME", "~/.modelpicker")).expanduser().resolve()
        saved_path = root / "settings.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.is_file() else {}
        return cls(
            data_dir=root,
            dataset_dir=Path(
                os.getenv("MODELPICKER_DATASET_DIR", saved.get("dataset_dir", str(root / "datasets")))
            )
            .expanduser()
            .resolve(),
            cache_dir=Path(os.getenv("MODELPICKER_CACHE_DIR", saved.get("cache_dir", str(root / "models"))))
            .expanduser()
            .resolve(),
            cpu_threads=int(os.getenv("MODELPICKER_CPU_THREADS", "2")),
            max_rows=int(os.getenv("MODELPICKER_MAX_ROWS", "1000")),
            max_import_rows=int(os.getenv("MODELPICKER_MAX_IMPORT_ROWS", "10000")),
            max_model_download_mb=int(os.getenv("MODELPICKER_MAX_MODEL_DOWNLOAD_MB", "1500")),
            min_available_ram_mb=int(os.getenv("MODELPICKER_MIN_AVAILABLE_RAM_MB", "1024")),
            max_worker_ram_mb=int(os.getenv("MODELPICKER_MAX_WORKER_RAM_MB", "4096")),
            max_job_seconds=int(os.getenv("MODELPICKER_MAX_JOB_SECONDS", "3600")),
        )

    def prepare(self) -> None:
        if not 1 <= self.cpu_threads <= 64:
            raise ValueError("CPU threads must be between 1 and 64")
        if (
            min(
                self.max_rows,
                self.max_import_rows,
                self.max_models,
                self.max_job_seconds,
                self.max_model_download_mb,
                self.min_available_ram_mb,
                self.max_worker_ram_mb,
            )
            <= 0
        ):
            raise ValueError("Resource limits must be positive")
        if self.max_rows > self.max_import_rows:
            raise ValueError("max_rows cannot exceed max_import_rows")
        for path in (self.data_dir, self.dataset_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.data_dir.chmod(0o700)


def hf_token() -> str | None:
    token = os.getenv("HF_TOKEN", "").strip()
    return token if token and not token.startswith(("your_", "hf_your")) else None
