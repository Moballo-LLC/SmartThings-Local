"""Public, redacted exception types for smartthings-local."""

__all__ = [
    'AuthenticationError',
    'AuthorizationError',
    'BlockwiseError',
    'EndpointError',
    'MalformedMessageError',
    'ObserveError',
    'ProbeError',
    'SessionClosedError',
    'SessionError',
    'SessionTimeoutError',
    'SmartThingsLocalError',
]


class SmartThingsLocalError(Exception):
    """Base class for classified library failures.

    Subclasses expose a stable ``code`` and a fixed, non-sensitive message.
    They intentionally do not accept arbitrary detail because backend errors
    can contain remote endpoints, local paths, or credential metadata.
    """

    code = 'smartthings_local_error'
    message = 'SmartThings Local operation failed'

    def __init__(self):
        super().__init__(self.message)

    def __repr__(self):
        return f'{type(self).__name__}(code={self.code!r})'


class EndpointError(SmartThingsLocalError, OSError):
    """An endpoint could not be resolved, bound, or connected."""

    code = 'endpoint'
    message = 'endpoint operation failed'


class ProbeError(SmartThingsLocalError, ConnectionError):
    """A DTLS probe failed before producing a protocol result."""

    code = 'probe'
    message = 'DTLS probe failed'


class SessionError(SmartThingsLocalError, ConnectionError):
    """A connected-session operation failed."""

    code = 'session'
    message = 'session operation failed'


class AuthenticationError(SessionError):
    """The peer or local credentials could not be authenticated."""

    code = 'authentication'
    message = 'authentication failed'


class AuthorizationError(SmartThingsLocalError, PermissionError):
    """The authenticated peer is not authorized for an operation."""

    code = 'authorization'
    message = 'operation is not authorized'


class SessionTimeoutError(SmartThingsLocalError, TimeoutError):
    """A bounded session operation exceeded its deadline."""

    code = 'timeout'
    message = 'session operation timed out'


class SessionClosedError(SessionError):
    """An operation was attempted on a closed session."""

    code = 'session_closed'
    message = 'session is closed'


class MalformedMessageError(SmartThingsLocalError, ValueError):
    """A protocol message could not be decoded safely."""

    code = 'malformed_message'
    message = 'malformed protocol message'


class BlockwiseError(SessionError):
    """A Block1 or Block2 transfer violated its bounded contract."""

    code = 'blockwise'
    message = 'blockwise transfer failed'


class ObserveError(SessionError):
    """A CoAP Observe relation could not be established or maintained."""

    code = 'observe'
    message = 'Observe relation failed'
