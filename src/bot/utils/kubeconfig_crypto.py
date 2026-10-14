"""Encrypt/decrypt kubeconfigs at rest using AES-GCM.

Kubeconfigs in /root/.holmes/kubeconfig-only-admin-can-run/ hold seed cluster
admin creds. Encrypting them at rest prevents Holmes tools (which can run
shell commands) from casually `cat`-ing a usable kubeconfig and exfiltrating
the credentials back to the LLM.

We use Python's `cryptography` library rather than `gpg` so the container
doesn't need gpg-agent (which is not bundled with the default gpg package
and was failing every encrypt/decrypt call).

File layout (binary, all big-endian):
  [16-byte salt][12-byte nonce][N-byte AES-GCM ciphertext + 16-byte tag]

Derivation: scrypt(passphrase, salt, n=2**14, r=8, p=1, dklen=32) → 256-bit
key. The salt is randomly generated per file at encrypt time and stored in
the file header, so two encryptions of the same plaintext produce different
ciphertext (defense against known-plaintext correlation).

Threat model: the passphrase lives in this module's source. A determined
adversary with shell access to the bot container can still decrypt. This
protects against accidental/casual exfil via `cat`, not against targeted attack.
"""
import logging
import os
import shutil
import tempfile

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

logger = logging.getLogger(__name__)

PASSPHRASE = b"792002"
SALT_LEN = 16
NONCE_LEN = 12
KEY_LEN = 32
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1


def _derive_key(salt: bytes) -> bytes:
    kdf = Scrypt(
        salt=salt,
        length=KEY_LEN,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
    )
    return kdf.derive(PASSPHRASE)


def is_encrypted(path: str) -> bool:
    """True if the path has a .enc extension (i.e., already encrypted)."""
    return path.endswith(".enc")


def encrypt_file(plain_path: str) -> str:
    """Encrypt a plaintext file at `plain_path` to `plain_path + ".enc"`.

    On success: shreds the original plaintext, returns the .enc path.
    On failure: logs and returns the original `plain_path` unchanged so the
    operator notices the file wasn't migrated (rather than silently dropping
    it from the glob list in /reconcile).
    """
    plain_path = str(plain_path)
    enc_path = plain_path + ".enc"

    try:
        with open(plain_path, "rb") as f:
            plaintext = f.read()

        salt = os.urandom(SALT_LEN)
        nonce = os.urandom(NONCE_LEN)
        key = _derive_key(salt)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data=None)

        # Write atomically: write to enc_path.tmp, fsync, rename. This avoids
        # leaving a half-written .enc file if the process dies mid-write.
        tmp_enc = enc_path + ".tmp"
        with open(tmp_enc, "wb") as f:
            f.write(salt)
            f.write(nonce)
            f.write(ciphertext)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_enc, enc_path)

        # Shred the plaintext so it's not recoverable from disk. We use a
        # simple overwrite+unlink (not `shred` CLI) to stay cross-platform.
        _secure_delete(plain_path)
        return enc_path
    except Exception as e:
        logger.error("kubeconfig encryption failed: path=%s error=%s", plain_path, e)
        return plain_path


def decrypt_to_tempfile(enc_path: str) -> str | None:
    """Decrypt `enc_path` to a fresh 0600 tempfile under /tmp.

    Returns the tempfile path on success, or None on failure (wrong
    passphrase, corrupted file, missing file). Caller is responsible
    for calling cleanup_tempfile() on the returned path.
    """
    enc_path = str(enc_path)
    if not os.path.exists(enc_path):
        logger.error("kubeconfig decryption failed: path=%s not found", enc_path)
        return None

    tmp_dir = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
    fd, tmp_path = tempfile.mkstemp(prefix="kc-", dir=tmp_dir)
    try:
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

    try:
        with open(enc_path, "rb") as f:
            blob = f.read()

        if len(blob) < SALT_LEN + NONCE_LEN + 16:
            raise ValueError("encrypted file too short / corrupted")
        salt = blob[:SALT_LEN]
        nonce = blob[SALT_LEN:SALT_LEN + NONCE_LEN]
        ciphertext = blob[SALT_LEN + NONCE_LEN:]
        key = _derive_key(salt)
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, associated_data=None)

        with open(tmp_path, "wb") as f:
            f.write(plaintext)
            f.flush()
            os.fsync(f.fileno())
        return tmp_path
    except Exception as e:
        logger.error("kubeconfig decryption failed: path=%s error=%s", enc_path, e)
        cleanup_tempfile(tmp_path)
        return None


def cleanup_tempfile(path: str) -> None:
    """Securely delete and unlink a tempfile. Never raises (safe in `finally`)."""
    if not path:
        return
    try:
        if os.path.exists(path):
            _secure_delete(path)
    except Exception as e:
        logger.warning("tempfile cleanup failed: path=%s error=%s", path, e)


def _secure_delete(path: str) -> None:
    """Overwrite the file with random bytes (one pass) then unlink.

    Single-pass overwrite is sufficient on modern flash/SSD (multiple passes
    don't help and wear the disk). The data may persist in unused blocks but
    that requires root/disk-level forensics, out of our threat model.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "wb") as f:
            f.write(os.urandom(size))
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    os.unlink(path)


def migrate_kubeconfigs(kconfig_dir: str) -> None:
    """Encrypt every plaintext file in `kconfig_dir` (non-recursive).

    Idempotent: a second call is a no-op because all files already end in .enc.
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
