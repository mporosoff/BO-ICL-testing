"""Local atomic persistence without Windows MAX_PATH or long temporary basenames."""
import os
from pathlib import Path
import tempfile


def io_path(path):
    """Address the same absolute Windows path through its extended-length spelling."""
    value = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = (
            "\\\\?\\UNC\\" + value[2:]
            if value.startswith("\\\\")
            else "\\\\?\\" + value
        )
    return Path(value)


def atomic_bytes(path, value):
    path = io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".boicl-", suffix=".tmp", dir=path.parent
    )
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
