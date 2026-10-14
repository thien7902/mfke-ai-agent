"""Encrypt/decrypt kubeconfigs at rest using GPG symmetric.

Kubeconfigs in /root/.holmes/kubeconfig-only-admin-can-run/ hold seed cluster
admin creds. Encrypting them at rest prevents Holmes tools (which can run
shell commands) from casually `cat`-ing a usable kubeconfig and exfiltrating
the credentials back to the LLM.

Threat model: the passphrase lives in this module's source. A determined
adversary with shell access to the bot container can still decrypt. This
protects against accidental/casual exfil, not against targeted attack.
"""
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

GPG_PASSPHRASE = "792002"
GPG_BINARY = shutil.which("gpg") or "gpg"

# Tempfile directory: /tmp on the bot container (Linux); fall back to the
# platform default if /tmp doesn't exist (e.g., Windows test runs).
TMP_DIR = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()


def _run_gpg(args: list[str]) -> subprocess.CompletedProcess:
    """Run gpg with the standard batch flags, return the CompletedProcess.

    --pinentry-mode loopback is required on a server without gpg-agent
    installed — otherwise gpg tries to start the agent to prompt for the
    passphrase and fails ("failed to start agent"). Loopback makes gpg
    read the passphrase from the command-line flag directly.
    """
    cmd = [
        GPG_BINARY,
        "--batch",
        "--yes",
        "--pinentry-mode", "loopback",
        "--passphrase", GPG_PASSPHRASE,
        *args,
    ]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )


def is_encrypted(path: str) -> bool:
    """True if the path has a .gpg extension (i.e., already encrypted)."""
    return path.endswith(".gpg")


def encrypt_file(plain_path: str) -> str:
    """Encrypt a plaintext file at `plain_path` to `plain_path + ".gpg"`.

    On success: shreds the original plaintext, returns the .gpg path.
    On failure: logs and returns the original `plain_path` unchanged so the
    operator can see the file wasn't migrated (rather than silently dropping
    it from the glob list in /reconcile).
    """
    plain_path = str(plain_path)
    enc_path = plain_path + ".gpg"

    result = _run_gpg(
        ["--symmetric", "--cipher-algo", "AES256", "-o", enc_path, plain_path]
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        logger.error(
            "kubeconfig encryption failed: path=%s stderr=%s",
            plain_path, stderr,
        )
        return plain_path

    # Shred the plaintext so it's not recoverable from disk.
    subprocess.run(
        ["shred", "-u", plain_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return enc_path


def decrypt_to_tempfile(enc_path: str) -> str | None:
    """Decrypt `enc_path` to a fresh 0600 tempfile under /tmp.

    Returns the tempfile path on success, or None on failure (wrong
    passphrase, corrupted file, gpg missing, etc.). Caller is responsible
    for calling cleanup_tempfile() on the returned path.
    """
    enc_path = str(enc_path)
    fd, tmp_path = tempfile.mkstemp(prefix="kc-", dir=TMP_DIR)
    try:
        # fchmod isn't always available (Windows); fall back to chmod on the
        # path. mkstemp already creates with 0600 on POSIX, so this is mostly
        # belt-and-suspenders.
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
    finally:
        os.close(fd)
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass

    with open(tmp_path, "wb") as out:
        result = subprocess.run(
            [
                GPG_BINARY, "--batch", "--yes",
                "--pinentry-mode", "loopback",
                "--passphrase", GPG_PASSPHRASE,
                "--decrypt", enc_path,
            ],
            stdout=out,
            stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        logger.error(
            "kubeconfig decryption failed: path=%s stderr=%s",
            enc_path, stderr,
        )
        cleanup_tempfile(tmp_path)
        return None
    return tmp_path


def cleanup_tempfile(path: str) -> None:
    """Shred and unlink a tempfile. Never raises (safe in `finally` blocks)."""
    if not path:
        return
    try:
        subprocess.run(
            ["shred", "-u", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        logger.warning("tempfile cleanup failed: path=%s error=%s", path, e)


def migrate_kubeconfigs(kconfig_dir: str) -> None:
    """Encrypt every plaintext file in `kconfig_dir` (non-recursive).

    Idempotent: a second call is a no-op because all files already end in .gpg.
    Files that fail to encrypt are left as-is (operator notices via the glob
    pattern in /reconcile that the file wasn't picked up).
    """
    kconfig_dir = str(kconfig_dir)
    if not os.path.isdir(kconfig_dir):
        return

    for entry in sorted(os.listdir(kconfig_dir)):
        full = os.path.join(kconfig_dir, entry)
        if not os.path.isfile(full):
            continue
        if is_encrypted(full):
            continue
        # Skip dotfiles (e.g., editor swap files like .swp).
        if entry.startswith("."):
            continue
        encrypt_file(full)
