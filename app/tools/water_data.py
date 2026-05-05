"""water_data — USGS instantaneous water-data lookup.

Ported from reachy_mini_conversation_app/tools/water_data.py, dropping the
transcript_server map-card push.
"""

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


USGS_SITE_RDB = "https://waterservices.usgs.gov/nwis/site/"
USGS_IV_JSON = "https://waterservices.usgs.gov/nwis/iv/"

PARAM_LABELS: Dict[str, str] = {
    "00060": "Discharge (cfs)",
    "00065": "Gage height (ft)",
    "00010": "Water temperature (°C)",
    "00045": "Precipitation (in)",
    "00095": "Specific conductance (µS/cm)",
    "00300": "Dissolved oxygen (mg/L)",
    "00400": "pH",
    "63680": "Turbidity (FNU)",
}


def _http_get(url: str, timeout: int = 15) -> Tuple[int, str]:
    req = urllib.request.Request(
        url, headers={"User-Agent": "reachy-mini-jetson-assistant/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def _parse_rdb(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    header: Optional[List[str]] = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        if all(p and p[0].isdigit() for p in parts if p):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _find_site(state: str, name_search: str) -> Optional[Dict[str, str]]:
    params = {
        "format": "rdb",
        "stateCd": state.lower(),
        "siteStatus": "active",
        "hasDataTypeCd": "iv",
        "siteType": "ST",
        "siteOutput": "expanded",
    }
    url = f"{USGS_SITE_RDB}?{urllib.parse.urlencode(params)}"
    try:
        status, body = _http_get(url)
    except urllib.error.HTTPError as e:
        logger.warning("USGS site service HTTP %s: %s", e.code, e.reason)
        return None
    except Exception as e:
        logger.warning("USGS site service error: %s", e)
        return None

    if status != 200:
        return None

    needle = name_search.lower()
    matches: List[Dict[str, str]] = []
    for row in _parse_rdb(body):
        nm = row.get("station_nm", "").lower()
        if needle in nm:
            matches.append(row)

    if not matches:
        return None

    matches.sort(
        key=lambda r: (
            not r.get("station_nm", "").lower().startswith(needle),
            len(r.get("station_nm", "")),
        )
    )
    return matches[0]


def _fetch_iv(site_id: str) -> Optional[Dict[str, Any]]:
    params = {"format": "json", "sites": site_id, "period": "P1D"}
    url = f"{USGS_IV_JSON}?{urllib.parse.urlencode(params)}"
    try:
        status, body = _http_get(url)
        if status != 200:
            return None
        return json.loads(body)
    except Exception as e:
        logger.warning("USGS IV fetch error for %s: %s", site_id, e)
        return None


def _extract_latest(
    iv: Dict[str, Any]
) -> Tuple[str, Optional[float], Optional[float], List[Tuple[str, str, str]]]:
    series_list = iv.get("value", {}).get("timeSeries", []) or []
    station_name = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    readings: List[Tuple[str, str, str]] = []

    for ts in series_list:
        src = ts.get("sourceInfo", {}) or {}
        station_name = station_name or src.get("siteName", "")
        if lat is None or lon is None:
            geo = src.get("geoLocation", {}).get("geogLocation", {}) or {}
            try:
                lat = float(geo.get("latitude")) if geo.get("latitude") is not None else lat
                lon = float(geo.get("longitude")) if geo.get("longitude") is not None else lon
            except (TypeError, ValueError):
                pass

        var = ts.get("variable", {}) or {}
        code = (var.get("variableCode") or [{}])[0].get("value", "")
        label = PARAM_LABELS.get(code, var.get("variableName", code))

        values = (ts.get("values") or [{}])[0].get("value") or []
        if not values:
            continue
        latest = values[-1]
        v = latest.get("value", "")
        when = latest.get("dateTime", "")
        try:
            if float(v) <= -999998.0:
                continue
        except (TypeError, ValueError):
            pass
        readings.append((label, str(v), when))

    return station_name, lat, lon, readings


class WaterData(Tool):
    """USGS real-time stream/river water-data lookup."""

    name = "water_data"
    description = (
        "Look up real-time USGS river/stream data for a US monitoring station. "
        "Provide either an 8-digit `site_id`, OR both `state` (2-letter US "
        "state code) and `name_search` (substring of the river/station name). "
        "Returns the station name, latest gage height / discharge / water "
        "temperature when available, so you can speak the result."
    )
    example = "What's the river level for the Mississippi in Missouri?"
    parameters_schema = {
        "type": "object",
        "properties": {
            "site_id": {
                "type": "string",
                "description": "8-digit USGS site number (e.g. '05536123').",
            },
            "state": {
                "type": "string",
                "description": "Two-letter US state code (required without site_id).",
            },
            "name_search": {
                "type": "string",
                "description": "Substring of station/river name (required without site_id).",
            },
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        site_id: str = (kwargs.get("site_id") or "").strip()
        state: str = (kwargs.get("state") or "").strip()
        name_search: str = (kwargs.get("name_search") or "").strip()

        logger.info(
            "Tool call: water_data site_id=%r state=%r name_search=%r",
            site_id, state, name_search,
        )

        site_meta: Optional[Dict[str, str]] = None
        if not site_id:
            if not state or not name_search:
                return {"error": "Provide either site_id, or both state and name_search."}
            site_meta = await asyncio.to_thread(_find_site, state, name_search)
            if not site_meta:
                return {"error": f"No active USGS gauge in {state.upper()} matching '{name_search}'."}
            site_id = site_meta.get("site_no", "").strip()
            if not site_id:
                return {"error": "Resolved site has no site_no."}

        iv = await asyncio.to_thread(_fetch_iv, site_id)
        if not iv:
            return {"error": f"Failed to fetch USGS data for site {site_id}."}

        station_name, lat, lon, readings = _extract_latest(iv)

        if (lat is None or lon is None) and site_meta:
            try:
                lat = float(site_meta.get("dec_lat_va", "")) if site_meta.get("dec_lat_va") else lat
                lon = float(site_meta.get("dec_long_va", "")) if site_meta.get("dec_long_va") else lon
            except ValueError:
                pass
        if not station_name and site_meta:
            station_name = site_meta.get("station_nm", station_name)

        if deps.broadcaster is not None and lat is not None and lon is not None:
            popup = f"<b>{station_name}</b><br>USGS {site_id}"
            if readings:
                popup += "<br>" + "<br>".join(
                    f"{lbl}: {val}" for lbl, val, _ in readings[:4]
                )
            try:
                deps.broadcaster.send({
                    "type": "map_card",
                    "title": f"USGS {site_id} – {station_name}",
                    "markers": [{"lat": lat, "lon": lon, "popup": popup}],
                    "zoom": 12,
                    "values": [[lbl, val] for lbl, val, _ in readings[:6]],
                })
            except Exception as e:
                logger.warning("Failed to broadcast water-data map card: %s", e)

        if not readings:
            summary = (
                f"Found USGS station {site_id} ({station_name}) but no current "
                f"readings are available."
            )
        else:
            top = ", ".join(f"{lbl} {val}" for lbl, val, _ in readings[:3])
            when = readings[0][2]
            summary = f"USGS {site_id}, {station_name}: {top}. Latest reading at {when}."

        return {
            "status": "ok",
            "site_id": site_id,
            "station_name": station_name,
            "latitude": lat,
            "longitude": lon,
            "readings": [
                {"label": lbl, "value": val, "time": when}
                for lbl, val, when in readings
            ],
            "summary": summary,
        }
