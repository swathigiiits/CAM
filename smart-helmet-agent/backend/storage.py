"""Persistent session storage (JSON + upload files, with optional S3 sync)."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
SESSIONS_FILE = DATA_DIR / "sessions.json"

S3_BUCKET = os.environ.get("S3_BUCKET", "")


def _s3():
    """Return boto3 S3 client, or None if not configured."""
    if not S3_BUCKET:
        return None, None
    try:
        import boto3
        client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        return client, S3_BUCKET
    except Exception:
        return None, None


def _s3_upload_file(local_path: Path, s3_key: str) -> None:
    client, bucket = _s3()
    if client is None:
        return
    try:
        client.upload_file(str(local_path), bucket, s3_key)
    except Exception as e:
        print(f"[S3] upload failed for {s3_key}: {e}")


def _s3_upload_bytes(data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> None:
    client, bucket = _s3()
    if client is None:
        return
    try:
        import io
        client.upload_fileobj(io.BytesIO(data), bucket, s3_key, ExtraArgs={"ContentType": content_type})
    except Exception as e:
        print(f"[S3] upload failed for {s3_key}: {e}")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if not SESSIONS_FILE.exists():
        SESSIONS_FILE.write_text("{}", encoding="utf-8")


def session_dir(session_id: str) -> Path:
    d = UPLOADS_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_all() -> Dict[str, Any]:
    ensure_dirs()
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_all(sessions: Dict[str, Any]) -> None:
    ensure_dirs()
    content = json.dumps(sessions, default=str)
    SESSIONS_FILE.write_text(content, encoding="utf-8")
    _s3_upload_bytes(content.encode(), "sessions/sessions.json", "application/json")


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return load_all().get(session_id)


def save_session(session_id: str, data: Dict[str, Any]) -> None:
    sessions = load_all()
    sessions[session_id] = data
    save_all(sessions)


def delete_session(session_id: str) -> None:
    sessions = load_all()
    sessions.pop(session_id, None)
    save_all(sessions)
