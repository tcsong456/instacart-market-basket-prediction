from instacart_etl_rnn.gold.create_product_training_data import build_word_idx


def test_build_word_idx_assigns_indices_by_frequency_and_word(
    spark,
):
    products = spark.createDataFrame(
        [
            (1, "Apple Banana"),
            (2, "apple   carrot"),
            (3, "BANANA carrot"),
            (4, "apple durian"),
        ],
        ["product_id", "product_name"],
    )

    result = build_word_idx(
        products,
        min_word_freq=2,
    )

    actual = {row["word"]: row["word_idx"] for row in result.collect()}

    assert actual == {
        "apple": 1,
        "banana": 2,
        "carrot": 3,
        "durian": 0,
    }


def test_build_word_idx_normalizes_case_and_whitespace(
    spark,
):
    products = spark.createDataFrame(
        [
            (1, "  Organic   Milk  "),
            (2, "ORGANIC milk"),
        ],
        ["product_id", "product_name"],
    )

    result = build_word_idx(
        products,
        min_word_freq=1,
    )

    actual = {row["word"]: row["word_idx"] for row in result.collect()}

    assert set(actual) == {
        "organic",
        "milk",
    }


def test_build_word_idx_assigns_zero_to_words_below_threshold(
    spark,
):
    products = spark.createDataFrame(
        [
            (1, "apple banana"),
            (2, "apple carrot"),
        ],
        ["product_id", "product_name"],
    )

    result = build_word_idx(
        products,
        min_word_freq=2,
    )

    actual = {row["word"]: row["word_idx"] for row in result.collect()}

    assert actual == {
        "apple": 1,
        "banana": 0,
        "carrot": 0,
    }
