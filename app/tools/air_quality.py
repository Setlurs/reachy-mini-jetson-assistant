"""air_quality — Open-Meteo air-quality lookup, keyless.

Ported from reachy_mini_conversation_app/tools/air_quality.py, dropping the
transcript_server map-card push (we don't have an equivalent UI surface yet).
"""

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
CURRENT_FIELDS = "us_aqi,pm10,pm2_5,carbon_monoxide,ozone,nitrogen_dioxide,sulphur_dioxide"

POLLUTANT_LABELS: List[Tuple[str, str]] = [
    ("us_aqi", "US AQI"),
    ("pm2_5", "PM2.5 (µg/m³)"),
    ("pm10", "PM10 (µg/m³)"),
    ("ozone", "Ozone (µg/m³)"),
    ("nitrogen_dioxide", "NO₂ (µg/m³)"),
    ("sulphur_dioxide", "SO₂ (µg/m³)"),
    ("carbon_monoxide", "CO (µg/m³)"),
]


def _aqi_category(aqi: Optional[float]) -> str:
    if aqi is None:
        return "Unknown"
    try:
        v = float(aqi)
    except (TypeError, ValueError):
        return "Unknown"
    if v <= 50:
        return "Good"
    if v <= 100:
        return "Moderate"
    if v <= 150:
        return "Unhealthy for Sensitive Groups"
    if v <= 200:
        return "Unhealthy"
    if v <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def _http_get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "reachy-mini-jetson-assistant/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _geocode(place: str) -> Optional[Dict[str, Any]]:
    params = {"name": place, "count": 1, "language": "en", "format": "json"}
    url = f"{GEOCODE_URL}?{urllib.parse.urlencode(params)}"
    try:
        data = json.loads(_http_get(url))
    except Exception as e:
        logger.warning("geocoding failed: %s", e)
        return None
    results = data.get("results") or []
    return results[0] if results else None


def _fetch_aq(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "current": CURRENT_FIELDS,
        "timezone": "auto",
    }
    url = f"{AQ_URL}?{urllib.parse.urlencode(params)}"
    try:
        return json.loads(_http_get(url))
    except Exception as e:
        logger.warning("air quality fetch failed: %s", e)
        return None


class AirQuality(Tool):
    """Look up current air quality (US AQI + pollutants) for a place."""

    name = "air_quality"
    description = (
        "Look up current air-quality readings (US AQI plus PM2.5, PM10, ozone, "
        "NO₂, SO₂, CO) for a place anywhere in the world. Provide either a "
        "`place` name (city/region) OR explicit `latitude` and `longitude`. "
        "Returns a brief spoken summary you should voice succinctly."
    )
    example = "What's the air quality in Los Angeles?"
    parameters_schema = {
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "Place name to geocode (e.g. 'San Francisco', 'Paris').",
            },
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        place: str = (kwargs.get("place") or "").strip()
        lat = kwargs.get("latitude")
        lon = kwargs.get("longitude")
        logger.info("Tool call: air_quality place=%r lat=%s lon=%s", place, lat, lon)

        label = place or ""
        admin: Optional[str] = None
        country: Optional[str] = None

        if lat is None or lon is None:
            if not place:
                return {"error": "Provide either `place`, or both `latitude` and `longitude`."}
            geo = await asyncio.to_thread(_geocode, place)
            if not geo:
                return {"error": f"Could not find a location for '{place}'."}
            lat = geo.get("latitude")
            lon = geo.get("longitude")
            label = geo.get("name") or place
            admin = geo.get("admin1")
            country = geo.get("country")

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            return {"error": "Invalid latitude/longitude."}

        aq = await asyncio.to_thread(_fetch_aq, lat_f, lon_f)
        if not aq or "current" not in aq:
            return {"error": f"Could not fetch air quality for {label}."}

        current = aq.get("current", {}) or {}
        units = aq.get("current_units", {}) or {}

        readings: List[Tuple[str, str, str]] = []
        for key, lbl in POLLUTANT_LABELS:
            v = current.get(key)
            if v is None:
                continue
            unit = units.get(key, "")
            readings.append((lbl, f"{v}", unit))

        us_aqi = current.get("us_aqi")
        category = _aqi_category(us_aqi)
        parts = [p for p in (label, admin, country) if p]
        place_label = ", ".join(dict.fromkeys(parts))

        if deps.broadcaster is not None:
            popup_lines = [f"<b>{place_label}</b>"]
            if us_aqi is not None:
                popup_lines.append(f"US AQI: {us_aqi} ({category})")
            popup_lines.append(f"PM2.5: {current.get('pm2_5', '–')} µg/m³")
            popup_lines.append(f"PM10: {current.get('pm10', '–')} µg/m³")
            try:
                deps.broadcaster.send({
                    "type": "map_card",
                    "title": f"Air quality – {place_label}",
                    "markers": [{
                        "lat": lat_f, "lon": lon_f,
                        "popup": "<br>".join(popup_lines),
                    }],
                    "zoom": 9,
                    "values": [[lbl, val] for lbl, val, _ in readings[:6]],
                })
            except Exception as e:
                logger.warning("Failed to broadcast air-quality map card: %s", e)

        if us_aqi is not None:
            summary = (
                f"Air quality in {place_label}: US AQI {int(us_aqi)} ({category}). "
                f"PM2.5 is {current.get('pm2_5', '–')} micrograms per cubic meter."
            )
        else:
            summary = f"Pulled air quality for {place_label}, but US AQI was not available."

        return {
            "status": "ok",
            "place": place_label,
            "latitude": lat_f,
            "longitude": lon_f,
            "us_aqi": us_aqi,
            "category": category,
            "time": current.get("time"),
            "readings": [
                {"label": lbl, "value": val, "unit": unit}
                for lbl, val, unit in readings
            ],
            "summary": summary,
        }
