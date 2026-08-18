from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from instacart_etl_rnn.common.utils import pad_array, parse_string_sequence


def build_word_idx(products: DataFrame, min_word_freq: int = 5) -> DataFrame:
    """Build a word-to-index vocabulary from product names.

    Product names are converted to lowercase, trimmed, and split on
    whitespace. Words occurring at least ``min_word_freq`` times are assigned
    positive integer indices ordered by descending frequency and then
    alphabetically for frequency ties. Less frequent words are assigned index
    0.

    Args:
        products: Spark DataFrame containing a ``product_name`` column.
        min_word_freq: Minimum number of occurrences required for a word to
            receive a positive vocabulary index.

    Returns:
        A Spark DataFrame with columns ``word`` and ``word_idx``. Frequent
        words have deterministic positive indices starting from 1, while rare
        words have index 0.
    """

    word_count = (
        products.select(
            F.explode(F.split(F.trim(F.lower("product_name")), r"\s+")).alias("word")
        )
        .filter(F.col("word") != "")
        .groupBy("word")
        .agg(F.count("*").alias("word_count"))
    )

    frequent_window = Window.orderBy(F.col("word_count").desc(), F.col("word").asc())
    frequent_words = (
        word_count.filter(F.col("word_count") >= min_word_freq)
        .withColumn("word_idx", F.row_number().over(frequent_window).cast("int"))
        .select("word", "word_idx")
    )

    rare_words = (
        word_count.filter(F.col("word_count") < min_word_freq)
        .withColumn("word_idx", F.lit(0).cast("int"))
        .select("word", "word_idx")
    )

    word_index = frequent_words.unionByName(rare_words)

    return word_index


def encode_product_names(products: DataFrame, word_index: DataFrame) -> DataFrame:
    """Encode product names as space-separated word index sequences.

    Product names are converted to lowercase, trimmed, split into tokens, and
    joined to the supplied word-index vocabulary. Token positions are preserved
    so that the encoded sequence follows the original word order.

    Products that do not produce any encoded tokens are assigned the fallback
    value ``"0"``.

    Args:
        products: Spark DataFrame containing ``product_id`` and ``product_name``.
        word_index: Spark DataFrame containing ``word`` and ``word_idx``.

    Returns:
        A Spark DataFrame containing ``product_id`` and
        ``product_name_encoded``, where ``product_name_encoded`` is a
        space-separated string of word indices.
    """

    cleaned_products = products.select(
        F.col("product_id").cast("int"),
        F.trim(F.lower("product_name")).alias("cleaned_product_name"),
    )

    product_tokens = (
        cleaned_products.select(
            "product_id",
            F.posexplode(F.split("cleaned_product_name", r"\s+")).alias(
                "word_pos", "word"
            ),
        )
        .filter(F.col("word") != "")
        .join(word_index, how="left", on="word")
        .groupBy("product_id")
        .agg(
            F.array_sort(F.collect_list(F.struct("word_pos", "word_idx"))).alias(
                "encoded_words"
            )
        )
        .select(
            "product_id",
            F.expr(
                """
                    array_join(
                        transform(
                            encoded_words,
                            x -> x.word_idx
                        ),
                        " "
                    )
                """
            ).alias("product_name_encoded"),
        )
    )

    result = (
        cleaned_products.select("product_id")
        .join(product_tokens, how="left", on="product_id")
        .withColumn(
            "product_name_encoded",
            F.coalesce(F.col("product_name_encoded"), F.lit("0")),
        )
    )

    return result


def build_product_training_data(
    product_history_data: DataFrame,
    encoded_product_name: DataFrame,
    *,
    product_name_length: int,
    encode_length: int,
) -> DataFrame:
    """Build fixed-length product training features.

    Joins encoded product-name data onto product history data, converts
    space-separated sequence columns into integer arrays, and pads or truncates
    those arrays to fixed lengths suitable for model training.

    Product-name sequences are padded or truncated to ``product_name_length``.
    Historical sequence columns are padded or truncated to ``encode_length``.
    The effective sequence lengths are stored in ``product_name_length`` and
    ``history_length``.

    Missing encoded product names are treated as empty sequences and therefore
    padded entirely with zeros.

    Args:
        product_history_data: Spark DataFrame containing product-level history
            features as space-separated string sequences.
        encoded_product_name: Spark DataFrame containing ``product_id`` and
            ``product_name_encoded``.
        product_name_length: Maximum length of the encoded product-name
            sequence.
        encode_length: Maximum length of historical feature sequences.

    Returns:
        A Spark DataFrame containing fixed-length integer-array features and
        their corresponding sequence lengths.
    """

    history_columns = [
        "position_in_order_history",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
        "history_order_size",
        "history_reorder_size",
        "order_numbers",
    ]

    df = (
        product_history_data.join(
            encoded_product_name,
            how="left",
            on="product_id",
        )
        .withColumn(
            "product_name_encoded",
            F.coalesce("product_name_encoded", F.lit("")),
        )
        .withColumn(
            "_parsed_product_name",
            parse_string_sequence(F.col("product_name_encoded")),
        )
    )

    padded_product_name, product_name_seq_length = pad_array(
        F.col("_parsed_product_name"),
        product_name_length,
    )

    df = df.withColumn(
        "product_name_encoded",
        padded_product_name,
    ).withColumn(
        "product_name_length",
        product_name_seq_length.cast("int"),
    )

    parsed_is_ordered_history = parse_string_sequence(F.col("is_ordered_history"))

    padded_is_ordered_history, history_length = pad_array(
        parsed_is_ordered_history,
        encode_length,
    )

    df = df.withColumn(
        "is_ordered_history",
        padded_is_ordered_history,
    ).withColumn(
        "history_length",
        history_length.cast("int"),
    )

    for colname in history_columns:
        if colname == "days_since_prior_orders":
            parsed_col = parse_string_sequence(F.col(colname), data_type="double")
            padded_col, _ = pad_array(parsed_col, encode_length, data_type="double")
        else:
            parsed_col = parse_string_sequence(F.col(colname))
            padded_col, _ = pad_array(parsed_col, encode_length)

        df = df.withColumn(
            colname,
            padded_col,
        )

    return df.select(
        "user_id",
        "product_id",
        "train_eval_set",
        "label",
        "product_name_encoded",
        "is_ordered_history",
        "position_in_order_history",
        "history_order_size",
        "history_reorder_size",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
        "order_numbers",
        "history_length",
        "product_name_length",
    )
