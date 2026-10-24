import pytest

from celal.config import load_settings
from celal.paths import ModelDiskMissing, configure_model_storage


def test_missing_model_disk_is_a_clear_turkish_error(tmp_path):
    env = {"CELAL_MODELS_DIR": str(tmp_path / "KIOXIA" / "Modeller")}
    with pytest.raises(ModelDiskMissing, match="model diski takılı değil"):
        configure_model_storage(env)
    assert "HF_HOME" not in env  # nothing points anywhere → nothing downloads to the internal disk


def test_model_caches_point_to_models_dir(tmp_path):
    env = {"CELAL_MODELS_DIR": str(tmp_path)}
    root = configure_model_storage(env)
    assert root == tmp_path
    assert env["HF_HOME"] == str(tmp_path / "huggingface")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "huggingface" / "hub")
    assert env["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    assert env["TORCH_HOME"] == str(tmp_path / "torch")
    assert (tmp_path / "huggingface" / "hub").is_dir()


def test_settings_from_toml_and_env(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[ssh]\nhost = "u@h"\nkey = "~/.ssh/k"\n[hermes]\nlocal_port = 9000\nconversation = "c1"\n')
    s = load_settings(env={}, config_path=cfg)
    assert (s.ssh_host, s.ssh_key, s.local_port, s.conversation, s.api) == ("u@h", "~/.ssh/k", 9000, "c1", "responses")
    s = load_settings(env={"CELAL_PORT": "9100", "CELAL_API": "chat"}, config_path=cfg)
    assert s.local_port == 9100 and s.api == "chat" and s.base_url == "http://127.0.0.1:9100"


def test_settings_without_config_file(tmp_path):
    s = load_settings(env={}, config_path=tmp_path / "yok.toml")
    assert s.ssh_host is None and s.local_port == 8642


def test_bad_api_mode(tmp_path):
    with pytest.raises(ValueError):
        load_settings(env={"CELAL_API": "x"}, config_path=tmp_path / "yok.toml")
