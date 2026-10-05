from pathlib import Path

def test_core_config_files_exist():
    assert Path("configs/classes.yaml").exists()
    assert Path("configs/disposal_rules.yaml").exists()
    assert Path("data/manifests/dataset_sources.csv").exists()
    assert Path("data/manifests/image_manifest.csv").exists()
