import pytest

from instacart_etl_rnn.validation.utils import is_number


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (1.5, True),
        (-3, True),
        (0, True),
        (True, False),
        (False, False),
        ("1", False),
        (None, False),
        ([1], False),
        ({}, False),
    ],
)
def test_is_number(value, expected):
    assert is_number(value) is expected
