def _build_range_message(
    column_name: str, min_value: int | float | None, max_value: int | float | None
) -> str:
    if min_value is not None and max_value is not None:
        return f"column '{column_name}' must be {min_value} and {max_value}"

    if min_value is not None:
        return f"column '{column_name}' must be at least {min_value}"

    if max_value is not None:
        return f"column '{column_name} must be at most {max_value}'"
