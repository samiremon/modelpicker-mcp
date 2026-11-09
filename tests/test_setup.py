import json
import os
import subprocess
import sys

from modelpicker.config import Settings


def test_setup_preserves_venv_and_persists_authorized_directory(settings, monkeypatch):
    env = {**os.environ, "MODELPICKER_HOME": str(settings.data_dir)}
    env.pop("MODELPICKER_DATASET_DIR", None)
    result = subprocess.run(
        [sys.executable, "-m", "modelpicker", "setup", "--dataset-dir", str(settings.dataset_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((settings.data_dir / "desktop-config.json").read_text())["mcpServers"]["modelpicker"]
    assert config["command"] == os.path.abspath(sys.executable)
    assert config["env"]["MODELPICKER_DATASET_DIR"] == str(settings.dataset_dir)
    monkeypatch.setenv("MODELPICKER_HOME", str(settings.data_dir))
    monkeypatch.delenv("MODELPICKER_DATASET_DIR", raising=False)
    assert Settings.from_env().dataset_dir == settings.dataset_dir
    secret = (settings.data_dir / "owner-password.txt").read_text()
    assert len(secret) >= 24
    assert secret not in result.stdout
    result = subprocess.run(
        [sys.executable, "-m", "modelpicker", "setup"], env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0
    assert (settings.data_dir / "owner-password.txt").read_text() == secret


def test_resource_configuration_enforces_owner_ceiling(engine):
    configured = engine.configure_resources(cpu_threads=1, max_rows=20, max_worker_ram_mb=1024)
    assert configured["max_rows"] == 20
    assert engine.settings.cpu_threads == 1
    engine.configure_resources(cpu_threads=1)
    assert engine.settings.max_rows == 20
    assert engine.settings.max_worker_ram_mb == 1024
    import pytest

    with pytest.raises(ValueError, match="owner"):
        engine.configure_resources(cpu_threads=1, max_rows=1001, max_worker_ram_mb=1024)
