"""Project-specific exception hierarchy.

All FraudLens errors inherit from `FraudLensException` so callers can catch
the whole domain with a single except clause. Each exception carries a human
readable `message` plus an optional `details` mapping for structured context
(ids, upstream error codes, retry hints) that can be attached to log records
or API responses without losing type information.
"""

from typing import Any


class FraudLensException(Exception):
    """Base class for all FraudLens errors.

    Args:
        message: Human-readable description of the error.
        details: Optional structured context (ids, codes, etc.) that callers
            can log or surface in API responses.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:
        if not self.details:
            return self.message
        return f"{self.message} | details={self.details}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exception for JSON logging or API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class TransactionNotFoundError(FraudLensException):
    """Raised when a transaction id cannot be located in the data store."""


class ModelNotLoadedError(FraudLensException):
    """Raised when scoring is attempted before the XGBoost artifact is loaded."""


class AgentExecutionError(FraudLensException):
    """Raised when a LangGraph agent fails during tool execution or routing."""


class RagQueryError(FraudLensException):
    """Raised when retrieval, reranking, or citation lookup fails."""


class LLMProviderError(FraudLensException):
    """Raised when an upstream LLM call fails (timeout, rate limit, bad output)."""
