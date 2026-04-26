"""Tool: IP geolocation and device-consistency check (mock)."""

from __future__ import annotations

import json

from langchain_core.tools import tool

_COUNTRIES = ["US", "TR", "DE", "CN", "BR"]
_ISPS = [
    "AS9121-Turk-Telekom",
    "AS15169-Google-Cloud",
    "AS16509-Amazon-AWS",
    "AS3320-Deutsche-Telekom",
    "AS60068-Datacenter-NL",
]


def _parse_ip_octets(ip_address: str) -> tuple[int, int]:
    """Return (x, y) from 192.168.x.y; fall back to (0, 0) on parse error."""
    try:
        parts = ip_address.split(".")
        return int(parts[-2]), int(parts[-1])
    except (IndexError, ValueError):
        return 0, 0


@tool
def get_geolocation_context(ip_address: str, device_id: str = "") -> str:
    """Check IP geolocation and device-IP consistency for fraud signals.

    Detects VPN/proxy usage, impossible travel, and mismatches between
    the IP country and the device's registered country.
    Call this when ip_address is available in the transaction.

    Args:
        ip_address: Source IP address from the transaction.
        device_id: Device fingerprint or device_fingerprint field (optional).

    Returns:
        JSON with ip_country, vpn_detected, impossible_travel,
        risk_signals list, and isp.
    """
    x, y = _parse_ip_octets(ip_address)

    impossible_travel = x % 7 == 0
    vpn_detected = x % 11 == 0
    ip_country = _COUNTRIES[x % 5]

    risk_signals: list[str] = []
    if impossible_travel:
        risk_signals.append("impossible_travel")
    if vpn_detected:
        risk_signals.append("vpn_or_proxy_detected")

    return json.dumps(
        {
            "ip_address": ip_address,
            "ip_country": ip_country,
            "vpn_detected": vpn_detected,
            "impossible_travel": impossible_travel,
            "risk_signals": risk_signals,
            "isp": _ISPS[x % len(_ISPS)],
            "ip_reputation_score": round((x % 100) / 100, 2),
        }
    )
