"""Small Home Assistant REST API helpers for development scripts."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


def call_service(
    base_url: str, token: str, domain: str, service: str, data: dict
) -> None:
    """Call a Home Assistant service."""
    request = Request(
        f"{base_url}/api/services/{domain}/{service}",
        data=json.dumps(data).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        if response.status >= 300:
            msg = f"Home Assistant returned HTTP {response.status}"
            raise RuntimeError(msg)


def set_boolean(base_url: str, token: str, entity_id: str, *, value: bool) -> None:
    """Set an input_boolean state."""
    call_service(
        base_url,
        token,
        "input_boolean",
        "turn_on" if value else "turn_off",
        {"entity_id": entity_id},
    )


def set_number(base_url: str, token: str, entity_id: str, value: float) -> None:
    """Set an input_number state."""
    call_service(
        base_url,
        token,
        "input_number",
        "set_value",
        {"entity_id": entity_id, "value": value},
    )
