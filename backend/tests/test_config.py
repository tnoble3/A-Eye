from pathlib import Path

from app.config import resolve_default_cnn_checkpoint_path


def test_resolve_default_cnn_checkpoint_path_for_local_repo_layout(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "backend" / "app" / "config.py"
    checkpoint_path = repo_root / "ml" / "artifacts" / "cnn_baseline" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()

    resolved_path = resolve_default_cnn_checkpoint_path(config_path)

    assert resolved_path == checkpoint_path


def test_resolve_default_cnn_checkpoint_path_for_container_layout(tmp_path: Path):
    app_root = tmp_path / "app"
    config_path = app_root / "app" / "config.py"
    checkpoint_path = app_root / "ml" / "artifacts" / "cnn_baseline" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()

    resolved_path = resolve_default_cnn_checkpoint_path(config_path)

    assert resolved_path == checkpoint_path
