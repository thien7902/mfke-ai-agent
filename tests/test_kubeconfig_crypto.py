"""Tests for kubeconfig_crypto — gpg encrypt/decrypt roundtrip, migration, cleanup."""
import os
import shutil
import subprocess

import pytest

from src.bot.utils import kubeconfig_crypto


pytestmark = pytest.mark.skipif(
    shutil.which("gpg") is None,
    reason="gpg binary not available in this environment",
)


@pytest.fixture
def kc_dir(tmp_path):
    """Empty kubeconfig dir, ready for files."""
    d = tmp_path / "kconfigs"
    d.mkdir()
    return d


def _write_kubeconfig(path, name: str = "test-kc") -> str:
    """Write a fake kubeconfig YAML to path/name and return full path."""
    full = path / name
    full.write_text(
        f"apiVersion: v1\n"
        f"kind: Config\n"
        f"clusters:\n"
        f"- name: {name}\n"
        f"  cluster:\n"
        f"    server: https://example.com\n"
        f"contexts:\n"
        f"- name: garden\n"
        f"  context:\n"
        f"    cluster: {name}\n"
        f"    user: admin\n"
        f"current-context: garden\n"
        f"users:\n"
        f"- name: admin\n"
        f"  user:\n"
        f"    token: fake-token-for-{name}\n"
    )
    return str(full)


def test_is_encrypted():
    assert kubeconfig_crypto.is_encrypted("foo.gpg")
    assert not kubeconfig_crypto.is_encrypted("foo")
    assert not kubeconfig_crypto.is_encrypted("foo.yaml")


def test_encrypt_decrypt_roundtrip(kc_dir):
    plain = _write_kubeconfig(kc_dir, "han-gardener")
    plaintext_bytes = open(plain, "rb").read()

    # Encrypt
    enc = kubeconfig_crypto.encrypt_file(plain)
    assert enc == plain + ".gpg"
    assert os.path.exists(enc)
    assert not os.path.exists(plain), "plaintext must be shredded"

    # Ciphertext should NOT contain the original token
    enc_bytes = open(enc, "rb").read()
    assert b"fake-token-for-han-gardener" not in enc_bytes

    # Decrypt to tempfile
    tmp = kubeconfig_crypto.decrypt_to_tempfile(enc)
    assert tmp is not None
    assert os.path.exists(tmp)
    # 0o600 only enforced on POSIX; on Windows the umask replaces it.
    if os.name == "posix":
        assert oct(os.stat(tmp).st_mode)[-3:] == "600"
    assert open(tmp, "rb").read() == plaintext_bytes

    # Cleanup
    kubeconfig_crypto.cleanup_tempfile(tmp)
    assert not os.path.exists(tmp)


def test_migrate_encrypts_plaintext_files(kc_dir):
    p1 = _write_kubeconfig(kc_dir, "han-gardener")
    p2 = _write_kubeconfig(kc_dir, "hoa-lac-new-gardener")
    pre_gpg = _write_kubeconfig(kc_dir, "already-encrypted.gpg")

    kubeconfig_crypto.migrate_kubeconfigs(str(kc_dir))

    assert os.path.exists(p1 + ".gpg")
    assert os.path.exists(p2 + ".gpg")
    assert not os.path.exists(p1)
    assert not os.path.exists(p2)
    # The .gpg we put there ourselves should remain (not get a .gpg.gpg)
    assert os.path.exists(pre_gpg)
    assert not os.path.exists(pre_gpg + ".gpg")


def test_migrate_idempotent(kc_dir):
    p1 = _write_kubeconfig(kc_dir, "han-gardener")
    kubeconfig_crypto.migrate_kubeconfigs(str(kc_dir))
    assert os.path.exists(p1 + ".gpg")

    # Second call: no exception, no double-encryption
    kubeconfig_crypto.migrate_kubeconfigs(str(kc_dir))
    assert os.path.exists(p1 + ".gpg")
    assert not os.path.exists(p1 + ".gpg.gpg")


def test_migrate_missing_dir_is_noop(tmp_path):
    # Non-existent dir: just returns, no crash
    kubeconfig_crypto.migrate_kubeconfigs(str(tmp_path / "does-not-exist"))


def test_cleanup_tempfile_missing_path_is_noop():
    # Path doesn't exist — should not raise
    kubeconfig_crypto.cleanup_tempfile("/tmp/does-not-exist-kc-xxxx")


def test_cleanup_tempfile_empty_string_is_noop():
    kubeconfig_crypto.cleanup_tempfile("")
