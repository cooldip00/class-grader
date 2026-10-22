"""Get a recording onto local disk (upload or link) and key results by file content."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

MAX_BYTES = 2 * 1024**3


class IngestError(RuntimeError):
    pass


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(link: str) -> str:
    """Download a Google Drive share link or a direct https URL into a temp dir.
    The caller deletes the returned file's directory when finished (see cleanup)."""
    url = urlparse(link.strip())
    if url.scheme != "https":
        raise IngestError("Please paste an https link (Google Drive share link or direct file URL).")
    tmp = Path(tempfile.mkdtemp(prefix="grader_"))
    out = tmp / "recording.mp4"
    try:
        if url.netloc.endswith("drive.google.com") or url.netloc.endswith("docs.google.com"):
            import gdown
            if not gdown.download(url=link.strip(), output=str(out), quiet=True):
                raise IngestError("Google Drive download failed. Is the link shared as 'Anyone with the link'?")
        else:
            import httpx
            with httpx.stream("GET", link.strip(), follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                size = 0
                with open(out, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise IngestError("Recording is larger than 2 GB.")
                        f.write(chunk)
    except IngestError:
        cleanup(str(out))
        raise
    except Exception as e:  # network / permission errors from gdown or httpx
        cleanup(str(out))
        raise IngestError(f"Could not download the recording: {e}") from e
    if out.stat().st_size < 10_000:
        cleanup(str(out))
        raise IngestError("The download is too small to be a recording (private link or a web page?).")
    return str(out)


def cleanup(path: str | None) -> None:
    """Delete a downloaded recording and its temp folder (uploads are removed by Gradio)."""
    if path and Path(path).parent.name.startswith("grader_"):
        shutil.rmtree(Path(path).parent, ignore_errors=True)
