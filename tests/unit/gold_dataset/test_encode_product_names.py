from instacart_etl_rnn.gold.create_product_training_data import encode_product_names


def test_encode_product_names_encodes_words_in_original_order(
    spark,
):
    products = spark.createDataFrame(
        [
            (1, "Organic Whole Milk"),
            (2, "BANANA Apple"),
        ],
        ["product_id", "product_name"],
    )

    word_index = spark.createDataFrame(
        [
            ("organic", 10),
            ("whole", 20),
            ("milk", 30),
            ("banana", 40),
            ("apple", 50),
        ],
        ["word", "word_idx"],
    )

    result = encode_product_names(products, word_index)

    actual = {
        row["product_id"]: row["product_name_encoded"] for row in result.collect()
    }

    assert actual == {
        1: "10 20 30",
        2: "40 50",
    }


def test_encode_product_names_normalizes_case_and_whitespace(
    spark,
):
    products = spark.createDataFrame(
        [
            (1, "   Organic    MILK   "),
        ],
        ["product_id", "product_name"],
    )

    word_index = spark.createDataFrame(
        [
            ("organic", 3),
            ("milk", 7),
        ],
        ["word", "word_idx"],
    )

    result = encode_product_names(products, word_index)

    row = result.first()

    assert row["product_id"] == 1
    assert row["product_name_encoded"] == "3 7"


def test_encode_product_names_uses_zero_when_product_has_no_tokens(
    spark,
):
    products = spark.createDataFrame(
        [
            (1, ""),
            (2, "apple"),
        ],
        ["product_id", "product_name"],
    )

    word_index = spark.createDataFrame(
        [
            ("apple", 5),
        ],
        ["word", "word_idx"],
    )

    result = encode_product_names(products, word_index)

    actual = {
        row["product_id"]: row["product_name_encoded"] for row in result.collect()
    }

    assert actual == {
        1: "0",
        2: "5",
    }
