"""Record entrypoint log uploads during tests.

Activated only when ``ENTRYPOINT_UPLOAD_SINK`` or ``ENTRYPOINT_UPLOAD_FAIL``
is set on the entrypoint process.
"""

import json
import os

_sink = os.environ.get("ENTRYPOINT_UPLOAD_SINK")
_fail = os.environ.get("ENTRYPOINT_UPLOAD_FAIL") == "1"

if _sink or _fail:
    import google.cloud.storage as storage

    class _Blob:
        def __init__(self, bucket_name: str, object_name: str) -> None:
            self._bucket_name = bucket_name
            self._object_name = object_name

        def upload_from_filename(
            self,
            filename: str,
            content_type: str | None = None,
        ) -> None:
            if _fail:
                raise RuntimeError("simulated upload failure")

            with open(filename, encoding="utf-8") as log_file:
                body = log_file.read()

            payload = {
                "uri": f"gs://{self._bucket_name}/{self._object_name}",
                "content_type": content_type,
                "body": body,
            }
            with open(_sink, "w", encoding="utf-8") as sink_file:
                json.dump(payload, sink_file)

    class _Bucket:
        def __init__(self, name: str) -> None:
            self._name = name

        def blob(self, object_name: str) -> _Blob:
            return _Blob(self._name, object_name)

    class _Client:
        def bucket(self, name: str) -> _Bucket:
            return _Bucket(name)

    storage.Client = _Client
