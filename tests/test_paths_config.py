import pytest

from celal.config import load_settings
from celal.paths import ModelDiskMissing, configure_model_storage


def test_default_models_dir_is_on_the_mac_and_created(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {}
    root = configure_model_storage(env)
    assert root == tmp_path / ".local" / "share" / "celal" / "models"
    assert (root / "huggingface" / "hub").is_dir() and "KIOXIA" not in env["HF_HUB_CACHE"]


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


def test_mac_settings(tmp_path):
    s = load_settings(env={}, config_path=tmp_path / "yok.toml")
    assert "Mac'te" in s.mac_prefixes and "bilgisayarda" in s.mac_prefixes
    assert (s.mac_cwd, s.mac_api, s.mac_api_port) == ("~", True, 8643)
    assert s.mac_api_key_file.endswith("mac-api-key")
    cfg = tmp_path / "config.toml"
    cfg.write_text('[mac]\nprefixes = ["Bilgisayar:"]\ncwd = "~/Kod"\napi = false\napi_port = 9643\ntimeout = 60\n')
    s = load_settings(env={}, config_path=cfg)
    assert (s.mac_prefixes, s.mac_cwd, s.mac_api, s.mac_api_port, s.mac_timeout) == (["Bilgisayar:"], "~/Kod", False, 9643, 60.0)
    s = load_settings(env={"CELAL_MAC_API": "1", "CELAL_MAC_PREFIXES": "a, b"}, config_path=cfg)
    assert s.mac_api is True and s.mac_prefixes == ["a", "b"]
    assert s.mac_max_parallel == 3  # default
    cfg.write_text("[mac]\nmax_parallel = 5\n")
    assert load_settings(env={}, config_path=cfg).mac_max_parallel == 5
    assert load_settings(env={"CELAL_MAC_MAX_PARALLEL": "2"}, config_path=cfg).mac_max_parallel == 2
    assert load_settings(env={"CELAL_MAC_MAX_PARALLEL": "0"}, config_path=cfg).mac_max_parallel == 1  # never below 1
