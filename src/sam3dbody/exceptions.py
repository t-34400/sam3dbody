"""Project-owned exception classes."""


class Sam3DBodyError(Exception):
    """Base exception for sam3dbody wrapper errors."""


class Sam3DBodyNotImplementedError(Sam3DBodyError, NotImplementedError):
    """Raised when a specified wrapper feature is not implemented yet."""
