"""IoTivity manufacturer-certificate OwnerPSK derivation contracts."""

from __future__ import annotations

import pytest

from smartthings_local.protocol.owner_psk import (
    CONFIRMED_MFG_CERTIFICATE_OXM_LABEL,
    MFG_CERTIFICATE_KEY_BLOCK_LENGTHS,
    STANDARD_MFG_CERTIFICATE_OXM_LABEL,
    derive_mfg_certificate_owner_psk,
)


_VALID_INPUTS = {
    "master_secret": bytes(range(48)),
    "client_random": bytes(range(32)),
    "server_random": bytes(range(32, 64)),
    "owner_uuid": bytes.fromhex("00112233445566778899aabbccddeeff"),
    "device_uuid": bytes.fromhex("ffeeddccbbaa99887766554433221100"),
    "cipher_name": "ECDHE-ECDSA-AES128-GCM-SHA256",
    "oxm_label": CONFIRMED_MFG_CERTIFICATE_OXM_LABEL,
}


def test_fixed_iotivity_gcm_vector():
    assert derive_mfg_certificate_owner_psk(**_VALID_INPUTS).hex() == (
        "ccd6c618a91290dee8c106544ed79a33"
    )


def test_owner_then_device_uuid_order_matches_iotivity_callers():
    reversed_context = derive_mfg_certificate_owner_psk(
        **{
            **_VALID_INPUTS,
            "owner_uuid": _VALID_INPUTS["device_uuid"],
            "device_uuid": _VALID_INPUTS["owner_uuid"],
        }
    )

    assert reversed_context.hex() == "8f0f2416c483546dc1806db769b21b68"
    assert reversed_context != derive_mfg_certificate_owner_psk(**_VALID_INPUTS)


def test_fixed_iotivity_ccm8_vector():
    inputs = {
        **_VALID_INPUTS,
        "cipher_name": "ECDHE-ECDSA-AES128-CCM8",
    }
    assert derive_mfg_certificate_owner_psk(**inputs).hex() == (
        "ddd3d945e266ee3dc27ff3a2c4321d32"
    )


def test_standard_and_confirmed_labels_derive_distinct_keys():
    confirmed = derive_mfg_certificate_owner_psk(**_VALID_INPUTS)
    standard = derive_mfg_certificate_owner_psk(
        **{**_VALID_INPUTS, "oxm_label": STANDARD_MFG_CERTIFICATE_OXM_LABEL}
    )

    assert standard.hex() == "26ee1fe4c3e74509a2f5db5ab41b1e47"
    assert standard != confirmed


def test_iotivity_cipher_key_block_lengths_are_immutable():
    assert dict(MFG_CERTIFICATE_KEY_BLOCK_LENGTHS) == {
        "ECDHE-ECDSA-AES128-SHA256": 96,
        "ECDHE-ECDSA-AES128-CCM": 40,
        "ECDHE-ECDSA-AES128-CCM8": 40,
        "ECDHE-ECDSA-AES128-GCM-SHA256": 120,
        "AES256-SHA256": 128,
        "ECDHE-ECDSA-AES256-SHA384": 160,
        "ECDHE-ECDSA-AES256-GCM-SHA384": 184,
        "AES128-GCM-SHA256": 120,
    }
    with pytest.raises(TypeError):
        MFG_CERTIFICATE_KEY_BLOCK_LENGTHS["new-cipher"] = 1


@pytest.mark.parametrize(
    ("field", "length"),
    [
        ("master_secret", 48),
        ("client_random", 32),
        ("server_random", 32),
        ("owner_uuid", 16),
        ("device_uuid", 16),
    ],
)
def test_binary_inputs_require_exact_bytes_and_lengths(field, length):
    with pytest.raises(TypeError, match=f"{field} must be bytes"):
        derive_mfg_certificate_owner_psk(
            **{**_VALID_INPUTS, field: bytearray(length)}
        )
    for invalid_length in (length - 1, length + 1):
        with pytest.raises(ValueError, match=f"exactly {length} bytes"):
            derive_mfg_certificate_owner_psk(
                **{**_VALID_INPUTS, field: b"x" * invalid_length}
            )


@pytest.mark.parametrize("field", ["owner_uuid", "device_uuid"])
def test_nil_uuid_is_rejected(field):
    with pytest.raises(ValueError, match="must not be the nil UUID"):
        derive_mfg_certificate_owner_psk(
            **{**_VALID_INPUTS, field: bytes(16)}
        )


def test_cipher_and_label_must_be_explicit_supported_values():
    with pytest.raises(TypeError, match="cipher_name must be a string"):
        derive_mfg_certificate_owner_psk(
            **{**_VALID_INPUTS, "cipher_name": b"cipher"}
        )
    with pytest.raises(ValueError, match="unexpected.*cipher"):
        derive_mfg_certificate_owner_psk(
            **{**_VALID_INPUTS, "cipher_name": "ECDHE-RSA-AES128-GCM-SHA256"}
        )
    with pytest.raises(TypeError, match="oxm_label must be bytes"):
        derive_mfg_certificate_owner_psk(
            **{**_VALID_INPUTS, "oxm_label": "oic.sec.doxm.mfgcert"}
        )
    with pytest.raises(ValueError, match="unexpected.*label"):
        derive_mfg_certificate_owner_psk(
            **{**_VALID_INPUTS, "oxm_label": b"unsupported"}
        )


def test_failures_do_not_include_key_material():
    key_material = b"private-master-secret"
    with pytest.raises(ValueError) as raised:
        derive_mfg_certificate_owner_psk(
            **{
                **_VALID_INPUTS,
                "master_secret": key_material,
                "cipher_name": "unsupported",
            }
        )
    assert key_material.hex() not in str(raised.value)
    assert "private-master-secret" not in str(raised.value)
