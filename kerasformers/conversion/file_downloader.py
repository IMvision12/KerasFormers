import hashlib
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """Validate if the provided URL is well-formed.

    Args:
        url: URL string to validate

    Returns:
        bool: True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_file(
    file_url: str, cache_dir: Optional[str] = None, force_download: bool = False
) -> str:
    """Download file from the specified URL.

    Streams the response to disk with a tqdm progress bar (via
    ``huggingface_hub``'s ``http_get``, the same downloader transformers uses),
    so no Keras / ``keras.utils.get_file`` dependency is involved. For
    ``huggingface.co`` URLs an ``HF_TOKEN`` in the environment is sent as a
    bearer token so gated / private repos resolve.

    Args:
        file_url: URL to download file from
        cache_dir: Directory to cache files (default: ~/.downloads)
        force_download: Force download even if file exists
    Returns:
        str: Path to the downloaded file
    Raises:
        ValueError: For invalid inputs
        Exception: For download failures
    """
    if not file_url:
        raise ValueError("file_url cannot be empty")
    if not validate_url(file_url):
        raise ValueError(f"Invalid URL format: {file_url}")

    cache_dir = Path(cache_dir or os.path.expanduser("~/.downloads"))

    file_name = os.path.basename(file_url)
    url_dir = file_url.rsplit("/", 1)[0]
    subdir = hashlib.sha1(url_dir.encode("utf-8")).hexdigest()[:16]
    target_dir = cache_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    local_file = target_dir / file_name

    if local_file.exists() and not force_download:
        print(f"Found cached file at {local_file}")
        return str(local_file)

    headers = {"User-Agent": "kerasformers"}
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in file_url:
        headers["Authorization"] = f"Bearer {token}"

    # Stream to a sibling ``.incomplete`` file, then atomically move it into
    # place, so an interrupted download never leaves a truncated file that a
    # later call would treat as cached.
    tmp_file = target_dir / f"{file_name}.incomplete"
    try:
        from huggingface_hub.file_download import http_get

        with open(tmp_file, "wb") as f:
            http_get(file_url, f, headers=headers)
        os.replace(tmp_file, local_file)
        return str(local_file)
    except Exception as e:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        print(f"Failed to download file: {str(e)}")
        raise
