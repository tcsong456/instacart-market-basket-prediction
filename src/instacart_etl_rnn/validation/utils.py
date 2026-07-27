from numbers import Integral, Number


def is_number(value: object) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def is_positive_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0
