from pathlib import Path

PathLike = Path | str


def _is_gcs_url(path: PathLike) -> bool:
    return str(path).startswith("gs://")


def join_path(base: PathLike, filename: str) -> PathLike:
    if isinstance(base, Path) and str(base).startswith("gs:"):
        raise ValueError("GCS paths must be provided as strings, not pathlib.Path.")

    if _is_gcs_url(base):
        return f"{str(base).rstrip('/')}/{filename.lstrip('/')}"
    return Path(base) / filename
