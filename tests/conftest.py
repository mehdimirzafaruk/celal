import pytest

from celal.config import Settings


@pytest.fixture
def settings(tmp_path):
    key = tmp_path / "api-key"
    key.write_text("test-secret-key\n")
    return Settings(
        ssh_host="user@example.invalid",
        ssh_key="~/.ssh/test",
        api_key_file=str(key),
        local_port=18642,
        mac_api=False,  # tests never bind 8643 or start a reverse ssh
        mac_api_key_file=str(tmp_path / "mac-api-key"),
        mac_cwd=str(tmp_path),
    )
