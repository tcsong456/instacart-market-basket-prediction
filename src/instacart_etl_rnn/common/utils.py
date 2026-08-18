from pyspark.sql import Column
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, FloatType, IntegerType


def parse_string_sequence(
    column: Column, pattern: str = r"\s+", data_type: str = "int"
) -> F.Column:
    """Parse a delimited string sequence into a Spark array.

    Splits a string column using the supplied pattern and casts each token
    to the requested Spark datatype. Null or blank input strings are converted
    to empty arrays of the requested element type.

    Args:
        column: Spark column containing the sequence to parse.
        pattern: Regular expression used to split the input string.
        data_type: Element datatype for the resulting array. Supported values
            are ``"int"``, ``"double"``, ``"float"``, and ``"bool"``.

    Returns:
        A Spark column containing an array of values of the requested datatype.

    Raises:
        ValueError: If ``data_type`` is not supported.
    """

    empty_array_types = {
        "int": IntegerType(),
        "double": DoubleType(),
        "float": FloatType(),
        "bool": BooleanType(),
    }

    cast_types = {
        "int": "int",
        "double": "double",
        "float": "float",
        "bool": "boolean",
    }

    element_type = empty_array_types.get(data_type)
    cast_type = cast_types.get(data_type)

    if element_type is None:
        raise ValueError(f"Data type: {data_type} is not supported!")
    empty_array = F.array().cast(ArrayType(element_type))

    return F.when(
        column.isNull() | (F.trim(column) == ""),
        empty_array,
    ).otherwise(
        F.transform(F.split(F.trim(column), pattern), lambda x: x.cast(cast_type))
    )


def pad_array(
    array_column: F.Column,
    max_length: int,
    data_type: str = "int",
) -> tuple[F.Column, F.Column]:
    """
    Truncate or pad an integer array to a fixed length.

    Args:
        array_column: Spark array column containing integer values.
        max_length: Desired output array length.

    Returns:
        A tuple containing:
            - padded_array: Array column of exactly ``max_length`` elements.
            - seq_length: Integer column representing the original sequence
              length after truncation (i.e. ``min(original_length, max_length)``).
    """

    truncated_array = F.slice(array_column, 1, max_length)
    seq_length = F.size(truncated_array).cast("int")
    padding_len = max_length - seq_length
    padded_array = F.concat(
        truncated_array, F.array_repeat(F.lit(0).cast(data_type), padding_len)
    )

    return padded_array, seq_length
