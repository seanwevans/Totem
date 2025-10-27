"""Cryptographic helpers for Totem."""
from __future__ import annotations

from ..constants import KEY_FILE, PUB_FILE

try:  # pragma: no cover - optional dependency
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature
except ImportError:  # pragma: no cover
    rsa = padding = hashes = serialization = default_backend = InvalidSignature = None

def ensure_keypair():  # pragma: no cover
    """Create an RSA keypair if it doesn't exist."""
    if rsa is None or serialization is None or default_backend is None:
        raise RuntimeError(
            "Cryptography support is unavailable; install the 'cryptography' package"
        )

    try:
        with open(KEY_FILE, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
    except FileNotFoundError:
        print("🔐 Generating new Totem RSA keypair ...")
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(KEY_FILE, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
        public_key = private_key.public_key()
        with open(PUB_FILE, "wb") as f:
            f.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
        print(f"  ✓ Keys written to {KEY_FILE}, {PUB_FILE}")
    return private_key


def sign_hash(sha256_hex):  # pragma: no cover
    """Sign a SHA256 hex digest with the private key."""
    if rsa is None or hashes is None or padding is None:
        raise RuntimeError(
            "Cryptography support is unavailable; install the 'cryptography' package"
        )

    private_key = ensure_keypair()
    signature = private_key.sign(
        sha256_hex.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    return signature.hex()


def verify_signature(sha256_hex, signature_hex):  # pragma: no cover
    """Verify a signature against the public key."""
    if (
        InvalidSignature is None
        or hashes is None
        or serialization is None
        or default_backend is None
        or padding is None
    ):
        raise RuntimeError(
            "Cryptography support is unavailable; install the 'cryptography' package"
        )

    with open(PUB_FILE, "rb") as f:
        public_key = serialization.load_pem_public_key(
            f.read(), backend=default_backend()
        )
    try:
        public_key.verify(
            bytes.fromhex(signature_hex),
            sha256_hex.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False




__all__ = [
    'ensure_keypair',
    'sign_hash',
    'verify_signature'
]
