from pathlib import Path

import yaml

import gtf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_is_importable() -> None:
    assert gtf.__version__ == "0.1.0"


def test_experiment_configs_are_valid_yaml() -> None:
    config_paths = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))

    assert config_paths
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert isinstance(config, dict)
        assert config
