from pathlib import Path
from src import config


def test_paths_under_project_root():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert (config.PROJECT_ROOT / "README.md").exists()
    assert config.CLEAN_ROOT == config.PROJECT_ROOT / "data" / "clean"
    assert config.RESULTS_ROOT == config.PROJECT_ROOT / "results"
    assert config.OUTPUTS_ROOT == config.PROJECT_ROOT / "outputs"


def test_dota_classes_count():
    assert len(config.DOTA_CLASSES) == 15
    assert config.DOTA_CLASSES[0] == "plane"


def test_seed_is_seven():
    assert config.SEED == 7


def test_subset_counts():
    assert config.N_SUBSET == 200
    assert config.N_TRAIN == 160
    assert config.TILE_SIZE == 1024
