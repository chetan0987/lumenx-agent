import os
import warnings
import requests
from typing import Optional
from urllib3.exceptions import InsecureRequestWarning

# Corporate SSL inspection proxies cause cert verification failures.
# Suppress the warning and disable verification for all outbound calls.
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

_BASE = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app")
_TOKEN = os.getenv("LUMENX_ADMIN_TOKEN", "")
_HEADERS = {"X-Admin-Token": _TOKEN}


def _get(path: str, params: dict = None) -> dict:
    resp = requests.get(f"{_BASE}{path}", headers=_HEADERS, params=params,
                        timeout=10, verify=False)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = requests.post(f"{_BASE}{path}",
                         headers={**_HEADERS, "Content-Type": "application/json"},
                         json=body, timeout=10, verify=False)
    resp.raise_for_status()
    return resp.json()


def get_inbox(since: Optional[str] = None) -> dict:
    """Return threads with unanswered customer messages.

    Pass `since` (ISO timestamp) from the previous response's server_time
    to avoid reprocessing the same threads.
    """
    params = {"since": since} if since else {}
    return _get("/api/admin/inbox", params=params)


def get_thread(thread_id: str) -> dict:
    """Return one thread with every message."""
    return _get(f"/api/admin/threads/{thread_id}")


def send_reply(thread_id: str, text: str,
               draft_source: str = "agent",
               confidence: Optional[float] = None) -> dict:
    """Send an admin reply on a thread."""
    body = {"text": text, "draft_source": draft_source}
    if confidence is not None:
        body["confidence"] = round(confidence, 4)
    return _post(f"/api/admin/threads/{thread_id}/reply", body)


def mark_read(thread_id: str) -> dict:
    """Clear the unread_admin counter on a thread."""
    return _post(f"/api/admin/threads/{thread_id}/mark-read", {})


def get_stats() -> dict:
    return _get("/api/admin/stats")


def get_products() -> dict:
    """Return all products with full detail plus company-wide policies."""
    return _get("/api/admin/products")


def get_export() -> dict:
    """Return full dump: every thread, every message, products, company policies."""
    return _get("/api/admin/export", params=None)
