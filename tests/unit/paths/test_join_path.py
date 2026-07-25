import pytest
from instacart_etl.common.paths import join_path
from pathlib import Path


@pytest.mark.parametrize(
    ('base_str', 'filename', 'expected'),
    [
        (
            'gs://my-bucket/raw',
            'orders.csv',
            'gs://my-bucket/raw/orders.csv'
        ),
        (
            'gs://my-bucket/raw/',
            'orders.csv',
            'gs://my-bucket/raw/orders.csv'
        ),
        (
            'gs://my-bucket',
            '/orders.csv',
            'gs://my-bucket/orders.csv'
        )
    ]
)
def test_gcs_join_path(
    base_str,
    filename,
    expected
):
    result = join_path(base_str, filename)
    assert result == expected
    assert isinstance(result, str)


@pytest.mark.parametrize(
    ('base_str', 'filename', 'expected'),
    [
        (
            Path('data/raw'),
            'orders.csv',
            Path('data/raw') / 'orders.csv'
        ),
        (
            'data/raw',
            'orders.csv',
            Path('data/raw') / 'orders.csv'
        )
    ]
)
def test_local_path_join(
    base_str,
    filename,
    expected
):
    result = join_path(base_str, filename)
    assert result == expected
    assert isinstance(result, Path)


@pytest.mark.parametrize(
    ('base_str', 'filename', 'expected'),
    [
        (
            '',
            'orders.csv',
            Path('orders.csv')
        ),
        (
            'data/raw',
            '',
            Path('data/raw')
        ),
        (
            '',
            '',
            Path('.')
        ),
        (
            'gs://my-bucket/raw',
            '',
            'gs://my-bucket/raw/'
        )
    ]
)
def test_edge_path_join(
    base_str,
    filename,
    expected
):
    result = join_path(base_str, filename)
    assert result == expected


def test_invalid_gcs_path():
    base_str = Path('gs://my-bucket')
    filename = ''
    with pytest.raises(ValueError, match='GCS paths must be provided'):
        join_path(base_str, filename)