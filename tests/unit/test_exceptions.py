"""Smoke tests for the FraudLens exception hierarchy."""

import pytest

from fraudlens.core.exceptions import (
    AgentExecutionError,
    FraudLensError,
    LLMProviderError,
    ModelNotLoadedError,
    RagQueryError,
    TransactionNotFoundError,
)


def test_base_exception_carries_message_and_empty_details() -> None:
    exc = FraudLensError("boom")
    assert exc.message == "boom"
    assert exc.details == {}
    assert str(exc) == "boom"


def test_base_exception_carries_details() -> None:
    exc = FraudLensError("boom", details={"txn_id": "abc", "code": 42})
    assert exc.details == {"txn_id": "abc", "code": 42}
    assert "txn_id" in str(exc)


def test_to_dict_is_json_serializable() -> None:
    exc = FraudLensError("boom", details={"k": "v"})
    payload = exc.to_dict()
    assert payload == {
        "error": "FraudLensError",
        "message": "boom",
        "details": {"k": "v"},
    }


@pytest.mark.parametrize(
    "exc_cls",
    [
        TransactionNotFoundError,
        ModelNotLoadedError,
        AgentExecutionError,
        RagQueryError,
        LLMProviderError,
    ],
)
def test_subclasses_inherit_from_base(exc_cls: type[FraudLensError]) -> None:
    exc = exc_cls("boom", details={"foo": "bar"})
    assert isinstance(exc, FraudLensError)
    assert exc.to_dict()["error"] == exc_cls.__name__
    assert exc.details == {"foo": "bar"}


def test_catching_base_catches_subclass() -> None:
    with pytest.raises(FraudLensError):
        raise TransactionNotFoundError("missing")
