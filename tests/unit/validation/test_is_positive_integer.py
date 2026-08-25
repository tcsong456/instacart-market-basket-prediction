import pytest

from instacart_etl_rnn.validation.utils import is_positive_integer


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (10, True),
        (100, True),
        (0, False),
        (-1, False),
        (-100, False),
        (1.0, False),
        (1.5, False),
        ("1", False),
        (None, False),
        (True, False),
        (False, False),
    ],
)
def test_is_positive_integer(value, expected):
    assert is_positive_integer(value) is expected
