"""Shared memory-BIO driver for bounded DTLS handshakes."""

from __future__ import annotations

import select
import time
from collections.abc import Callable

from OpenSSL import SSL

from .coap import split_dtls

_HANDSHAKE_POLL_S = 0.5
_MAX_DATAGRAM_SIZE = 65535


class _HandshakeCancelled(Exception):
    """Internal signal that a handshake wake socket became readable."""


def _drive_dtls_handshake(
    connection,
    sock,
    *,
    deadline: float,
    retries: int | None = None,
    on_datagram: Callable[[bytes], None] | None = None,
    wake_socket=None,
) -> bool:
    """Drive one memory-BIO DTLS handshake up to a monotonic deadline.

    OpenSSL owns the retransmission schedule. ``retries`` optionally limits
    how many expired retransmission timers are serviced; a normal session is
    bounded only by its deadline, while the diagnostic probe retains its
    explicit retry budget.

    Return ``True`` only when the handshake completes before the deadline.
    TLS and socket failures are left to the caller to classify.
    """
    retransmits = 0
    while time.monotonic() < deadline:
        try:
            connection.do_handshake()
            return time.monotonic() < deadline
        except SSL.WantReadError:
            pass

        try:
            output = connection.bio_read(_MAX_DATAGRAM_SIZE)
        except SSL.WantReadError:
            output = None
        if output:
            for record in split_dtls(output):
                if sock.send(record) != len(record):
                    raise OSError("incomplete UDP send")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait = min(_HANDSHAKE_POLL_S, remaining)
        timed_out = False
        if wake_socket is None:
            sock.settimeout(wait)
            try:
                datagram = sock.recv(_MAX_DATAGRAM_SIZE)
            except TimeoutError:
                timed_out = True
        else:
            readable, _, _ = select.select(
                (sock, wake_socket),
                (),
                (),
                wait,
            )
            if wake_socket in readable:
                raise _HandshakeCancelled()
            if sock not in readable:
                timed_out = True
            else:
                datagram = sock.recv(_MAX_DATAGRAM_SIZE)

        if timed_out:
            timer = connection.DTLSv1_get_timeout()
            if timer is not None and timer <= 0:
                if retries is not None and retransmits >= retries:
                    break
                connection.DTLSv1_handle_timeout()
                retransmits += 1
            continue

        if not datagram:
            continue
        if on_datagram is not None:
            on_datagram(datagram)
        connection.bio_write(datagram)

    return False
