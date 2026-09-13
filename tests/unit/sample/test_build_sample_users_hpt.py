from instacart_etl_rnn.sample.build_sample_users_hpt import build_hpt_user_sample


def test_build_hpt_user_sample_is_deterministic_and_unique(spark):
    df = spark.createDataFrame(
        [(user_id, product_id) for user_id in range(1, 21) for product_id in (1, 2)],
        ["user_id", "product_id"],
    )

    first = [row.user_id for row in build_hpt_user_sample(df).collect()]
    second = [row.user_id for row in build_hpt_user_sample(df).collect()]

    assert first == second
    assert len(first) == len(set(first))
    assert set(first) <= set(range(1, 21))
    assert build_hpt_user_sample(df).columns == ["user_id"]


def test_build_hpt_user_sample_keeps_about_thirty_percent_of_users(spark):
    user_ids = set(range(1, 201))
    df = spark.createDataFrame(
        [(user_id,) for user_id in user_ids],
        ["user_id"],
    )

    sampled = {row.user_id for row in build_hpt_user_sample(df).collect()}

    assert sampled
    assert sampled < user_ids
    assert 0.15 <= len(sampled) / len(user_ids) <= 0.45
