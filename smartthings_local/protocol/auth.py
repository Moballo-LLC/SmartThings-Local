"""Immutable authentication providers for DTLS sessions."""

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Protocol, runtime_checkable

from OpenSSL import SSL, _util, crypto

_OCF_ROOT_CA = str(Path(__file__).with_name("ocf_root_ca.pem"))
_DTLS_CIPHERS = b"ECDHE-ECDSA-AES128-GCM-SHA256:@SECLEVEL=0"
_DTLS_PSK_CIPHERS = b"ECDHE-PSK-AES128-CBC-SHA256:@SECLEVEL=0"
_PSK_CLIENT_CALLBACK_CDEF = (
    "unsigned int (*)(SSL *, char *, char *, unsigned int, "
    "unsigned char *, unsigned int)"
)
_PEM_CERT_RE = re.compile(
    rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
    re.DOTALL,
)


def _verify_peer(_connection, _certificate, _error, _depth, ok):
    """Keep pyOpenSSL's existing verification result unchanged."""
    return ok


def _load_pem_chain(ctx: SSL.Context, cert_pem: str, key_pem: str) -> None:
    """Load a PEM certificate chain and private key into a context in memory."""
    certificates = _PEM_CERT_RE.findall(cert_pem.encode())
    if not certificates:
        raise ValueError("No certificates found in cert_pem")
    ctx.use_certificate(
        crypto.load_certificate(crypto.FILETYPE_PEM, certificates[0])
    )
    for extra in certificates[1:]:
        ctx.add_extra_chain_cert(
            crypto.load_certificate(crypto.FILETYPE_PEM, extra)
        )
    ctx.use_privatekey(
        crypto.load_privatekey(crypto.FILETYPE_PEM, key_pem.encode())
    )
    ctx.check_privatekey()


@runtime_checkable
class AuthenticationProvider(Protocol):
    """Configure authentication for a newly created DTLS context."""

    def configure_context(self, context: SSL.Context) -> None:
        """Configure a context while this provider remains session-owned."""


class CertificateAuth:
    """Certificate authentication loaded from files or in-memory PEM data.

    Use :meth:`from_files` or :meth:`from_memory` to create an instance.
    Credential sources are intentionally not exposed as public attributes.
    """

    __slots__ = (
        "_certificate_path",
        "_certificate_pem",
        "_private_key_path",
        "_private_key_pem",
    )

    def __init__(
        self,
        *,
        certificate_path: str | PathLike[str] | None = None,
        private_key_path: str | PathLike[str] | None = None,
        certificate_pem: str | None = None,
        private_key_pem: str | None = None,
    ) -> None:
        file_supplied = (
            certificate_path is not None or private_key_path is not None
        )
        memory_supplied = (
            certificate_pem is not None or private_key_pem is not None
        )
        if file_supplied and memory_supplied:
            raise ValueError(
                "pass either certificate_path/private_key_path or "
                "certificate_pem/private_key_pem, not both"
            )
        if file_supplied:
            if certificate_path is None or private_key_path is None:
                raise ValueError(
                    "certificate_path and private_key_path must be passed together"
                )
        elif memory_supplied:
            if certificate_pem is None or private_key_pem is None:
                raise ValueError(
                    "certificate_pem and private_key_pem must be passed together"
                )
        else:
            raise ValueError(
                "must pass either certificate_path/private_key_path or "
                "certificate_pem/private_key_pem"
            )
        object.__setattr__(
            self,
            "_certificate_path",
            str(certificate_path) if certificate_path is not None else None,
        )
        object.__setattr__(
            self,
            "_private_key_path",
            str(private_key_path) if private_key_path is not None else None,
        )
        object.__setattr__(self, "_certificate_pem", certificate_pem)
        object.__setattr__(self, "_private_key_pem", private_key_pem)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CertificateAuth is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("CertificateAuth is immutable")

    @classmethod
    def from_files(
        cls,
        certificate_path: str | PathLike[str],
        private_key_path: str | PathLike[str],
    ) -> CertificateAuth:
        """Create a provider backed by certificate-chain and key files."""
        return cls(
            certificate_path=certificate_path,
            private_key_path=private_key_path,
        )

    @classmethod
    def from_memory(
        cls,
        certificate_pem: str,
        private_key_pem: str,
    ) -> CertificateAuth:
        """Create a provider backed by an in-memory PEM chain and key."""
        return cls(
            certificate_pem=certificate_pem,
            private_key_pem=private_key_pem,
        )

    def __repr__(self) -> str:
        """Return a representation that never includes credential material."""
        return "CertificateAuth()"

    def configure_context(self, context: SSL.Context) -> None:
        """Apply the existing certificate authentication profile to a context."""
        context.load_verify_locations(_OCF_ROOT_CA)
        context.set_verify(SSL.VERIFY_PEER, _verify_peer)
        # @SECLEVEL=0 permits SHA-1 in Samsung's server cert chain (AC14K_M
        # intermediate is SHA-1 signed). This is the only channel that reaches
        # the OpenSSL instance cryptography bundles; ctypes and cffi bindings
        # do not expose SSL_CTX_set_security_level on this build.
        context.set_cipher_list(_DTLS_CIPHERS)
        if self._certificate_pem is not None:
            _load_pem_chain(
                context,
                self._certificate_pem,
                self._private_key_pem,
            )
        else:
            context.use_certificate_chain_file(self._certificate_path)
            context.use_privatekey_file(self._private_key_path)
            context.check_privatekey()


class PskAuth:
    """DTLS authentication using an existing OCF PSK credential.

    The identity must be a raw 16-byte OCF UUID. The key must contain 16 or
    32 bytes. Credential material is intentionally not exposed as public
    attributes and is never included in this provider's representation. A
    configured context must not outlive this provider; ``DtlsCoapSession``
    enforces that lifetime by retaining its provider.
    """

    __slots__ = ("_callback",)

    def __init__(self, *, identity: bytes, key: bytes) -> None:
        if type(identity) is not bytes or type(key) is not bytes:
            raise TypeError("identity and key must be bytes")
        if len(identity) != 16:
            raise ValueError("identity must be a raw 16-byte OCF UUID")
        if b"\x00" in identity:
            raise ValueError("identity cannot contain a NUL byte")
        if len(key) not in (16, 32):
            raise ValueError("key must be 16 or 32 bytes")

        ffi = _util.ffi

        @ffi.callback(_PSK_CLIENT_CALLBACK_CDEF)
        def client_callback(
            _ssl,
            _identity_hint,
            identity_buffer,
            max_identity_length,
            key_buffer,
            max_key_length,
        ):
            # OpenSSL callbacks cannot propagate Python exceptions. Fail
            # before touching either destination when a buffer is unavailable
            # or too small for the complete credential.
            if (
                identity_buffer == ffi.NULL
                or key_buffer == ffi.NULL
                or len(identity) + 1 > max_identity_length
                or len(key) > max_key_length
            ):
                return 0
            ffi.memmove(
                identity_buffer,
                identity + b"\x00",
                len(identity) + 1,
            )
            ffi.memmove(key_buffer, key, len(key))
            return len(key)

        object.__setattr__(self, "_callback", client_callback)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PskAuth is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("PskAuth is immutable")

    def __repr__(self) -> str:
        """Return a representation that never includes credential material."""
        return "PskAuth()"

    def configure_context(self, context: SSL.Context) -> None:
        """Configure one context for the narrow Samsung OCF PSK profile."""
        setter = getattr(_util.lib, "SSL_CTX_set_psk_client_callback", None)
        if setter is None:
            raise RuntimeError(
                "the installed OpenSSL binding does not support DTLS PSK"
            )
        context.set_cipher_list(_DTLS_PSK_CIPHERS)
        setter(context._context, self._callback)  # noqa: SLF001


__all__ = ["AuthenticationProvider", "CertificateAuth", "PskAuth"]
