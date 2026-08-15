"""Pure IoTivity manufacturer-certificate OwnerPSK derivation."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
from types import MappingProxyType
from typing import Final


CONFIRMED_MFG_CERTIFICATE_OXM_LABEL: Final = b"x.org.iotivity.conmfgcert"
STANDARD_MFG_CERTIFICATE_OXM_LABEL: Final = b"oic.sec.doxm.mfgcert"

# OpenSSL cipher names mapped to the key-block lengths used by IoTivity's
# CAGenerateOwnerPSK implementation.
MFG_CERTIFICATE_KEY_BLOCK_LENGTHS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "ECDHE-ECDSA-AES128-SHA256": 96,
        "ECDHE-ECDSA-AES128-CCM": 40,
        "ECDHE-ECDSA-AES128-CCM8": 40,
        "ECDHE-ECDSA-AES128-GCM-SHA256": 120,
        "AES256-SHA256": 128,
        "ECDHE-ECDSA-AES256-SHA384": 160,
        "ECDHE-ECDSA-AES256-GCM-SHA384": 184,
        "AES128-GCM-SHA256": 120,
    }
)

_TLS_MASTER_SECRET_BYTES: Final = 48
_TLS_RANDOM_BYTES: Final = 32
_OCF_UUID_BYTES: Final = 16
_OWNER_PSK_BYTES: Final = 16


def _require_bytes(name: str, value: bytes, length: int) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != length:
        raise ValueError(f"{name} must be exactly {length} bytes")
    return value


def _require_uuid(name: str, value: bytes) -> bytes:
    value = _require_bytes(name, value, _OCF_UUID_BYTES)
    if not any(value):
        raise ValueError(f"{name} must not be the nil UUID")
    return value


def _tls12_p_hash_sha256(
    key: bytes,
    label: bytes,
    random1: bytes,
    random2: bytes,
    length: int,
) -> bytes:
    seed = label + random1 + random2
    a_value = hmac.new(key, seed, hashlib.sha256).digest()
    output = bytearray()
    while len(output) < length:
        output.extend(hmac.new(key, a_value + seed, hashlib.sha256).digest())
        a_value = hmac.new(key, a_value, hashlib.sha256).digest()
    return bytes(output[:length])


def derive_mfg_certificate_owner_psk(
    *,
    master_secret: bytes,
    client_random: bytes,
    server_random: bytes,
    owner_uuid: bytes,
    device_uuid: bytes,
    cipher_name: str,
    oxm_label: bytes,
) -> bytes:
    """Derive a 128-bit OwnerPSK from caller-supplied DTLS state.

    This implements IoTivity's two-stage TLS 1.2 SHA-256 P_hash operation.
    It performs no session access, network I/O, ownership writes, or storage.
    The caller must supply state from an authenticated manufacturer-certificate
    session and explicitly select the OXM label used by that transaction.
    """

    if not isinstance(cipher_name, str):
        raise TypeError("cipher_name must be a string")
    key_block_bytes = MFG_CERTIFICATE_KEY_BLOCK_LENGTHS.get(cipher_name)
    if key_block_bytes is None:
        raise ValueError("unexpected manufacturer-certificate DTLS cipher")

    master_secret = _require_bytes(
        "master_secret", master_secret, _TLS_MASTER_SECRET_BYTES
    )
    client_random = _require_bytes(
        "client_random", client_random, _TLS_RANDOM_BYTES
    )
    server_random = _require_bytes(
        "server_random", server_random, _TLS_RANDOM_BYTES
    )
    owner_uuid = _require_uuid("owner_uuid", owner_uuid)
    device_uuid = _require_uuid("device_uuid", device_uuid)
    if not isinstance(oxm_label, bytes):
        raise TypeError("oxm_label must be bytes")
    if oxm_label not in {
        CONFIRMED_MFG_CERTIFICATE_OXM_LABEL,
        STANDARD_MFG_CERTIFICATE_OXM_LABEL,
    }:
        raise ValueError("unexpected manufacturer-certificate OXM label")

    key_block = _tls12_p_hash_sha256(
        master_secret,
        b"key expansion",
        server_random,
        client_random,
        key_block_bytes,
    )
    # IoTivity's OTM callers pass the owner UUID first and the target device
    # UUID second. The lower adapter's historical rsrc/prov parameter names
    # describe those arguments inconsistently, so preserve the caller order.
    return _tls12_p_hash_sha256(
        key_block,
        oxm_label,
        owner_uuid,
        device_uuid,
        _OWNER_PSK_BYTES,
    )
