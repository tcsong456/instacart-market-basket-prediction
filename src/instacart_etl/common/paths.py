from pathlib import Path

PathLike = Path | str


def _is_gcs_url(path: PathLike) -> bool:
    return str(path).startswith("gs://")


def join_path(base: PathLike, filename: str) -> PathLike:
    if _is_gcs_url(base):
        return f"{str(base).rstrip('/')}/{filename}"
    return Path(base) / filename
