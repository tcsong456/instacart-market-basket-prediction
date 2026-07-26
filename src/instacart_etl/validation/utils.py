from numbers import Number


def is_number(value: object) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)
