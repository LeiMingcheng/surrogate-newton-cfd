"""Exceptions for the clean NK_resume runtime boundary."""

from __future__ import annotations


class NKResumeError(RuntimeError):
    """Base class for NK_resume errors."""


class NotMigratedError(NKResumeError):
    """Raised when a capability is intentionally not migrated yet."""

    def __init__(self, capability: str, *, detail: str | None = None) -> None:
        message = f"Capability is not migrated into NK_resume yet: {capability}"
        if detail:
            message = f"{message}. {detail}"
        super().__init__(message)


class ContractError(NKResumeError, ValueError):
    """Raised when a canonical case or plan violates the new contract."""


class LegacyBoundaryError(NKResumeError):
    """Raised when code attempts to cross into a legacy implementation path."""
