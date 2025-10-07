"""Round-trip tests for arithmetic codec (pending implementation)."""

import pytest


@pytest.mark.xfail(reason="Arithmetic codec not yet implemented", raises=NotImplementedError)
def test_arithmetic_roundtrip_pending() -> None:
    """Placeholder test until arithmetic codec is implemented."""

    pytest.skip("Pending arithmetic codec implementation")
