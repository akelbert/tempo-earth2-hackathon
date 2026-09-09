"""Bounded connectors to large public environmental data sources.

These helpers deliberately construct narrow requests.  They are escape hatches
from the curated event data, not invitations to copy entire public archives
onto a 20 GiB attendee volume.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

HRRR_ARCHIVE = "s3://noaa-hrrr-bdp-pds/"
GFS_ARCHIVE = "s3://noaa-gfs-bdp-pds/"
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
NOAA_COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
USER_AGENT = "tempo-earth2-workshop/1"


def _iso_day(value: str | date | datetime) -> str:
    if isinstance(value, str):
        return value[:10]
    return value.isoformat()[:10]


def usgs_instantaneous_values_url(
    sites: str | list[str],
    start: str | date | datetime,
    end: str | date | datetime,
    parameter_codes: str | list[str] = "00060",
) -> str:
    """Build a USGS JSON request (default parameter 00060 is discharge)."""
    if isinstance(sites, str):
        sites = [sites]
    if isinstance(parameter_codes, str):
        parameter_codes = [parameter_codes]
    query = {
        "format": "json",
        "sites": ",".join(sites),
        "startDT": _iso_day(start),
        "endDT": _iso_day(end),
        "parameterCd": ",".join(parameter_codes),
        "siteStatus": "all",
    }
    return f"{USGS_IV_URL}?{urlencode(query)}"


def read_usgs_instantaneous_values(
    sites: str | list[str],
    start: str | date | datetime,
    end: str | date | datetime,
    parameter_codes: str | list[str] = "00060",
) -> pd.DataFrame:
    """Read a bounded USGS request into a tidy table."""
    url = usgs_instantaneous_values_url(sites, start, end, parameter_codes)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    rows = []
    for series in payload["value"]["timeSeries"]:
        source = series["sourceInfo"]
        variable = series["variable"]
        site = source["siteCode"][0]["value"]
        for block in series["values"]:
            for item in block["value"]:
                rows.append(
                    {
                        "site": site,
                        "site_name": source.get("siteName", site),
                        "latitude": source["geoLocation"]["geogLocation"]["latitude"],
                        "longitude": source["geoLocation"]["geogLocation"]["longitude"],
                        "time_utc": item["dateTime"],
                        "parameter": variable["variableCode"][0]["value"],
                        "variable": variable.get("variableDescription", ""),
                        "value": item["value"],
                        "unit": variable.get("unit", {}).get("unitCode", ""),
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def noaa_coops_url(
    station: str,
    start: str | date | datetime,
    end: str | date | datetime,
    product: str = "water_level",
    datum: str = "MSL",
    units: str = "metric",
    interval: str = "h",
) -> str:
    """Build a bounded NOAA CO-OPS Data API request."""
    query = {
        "product": product,
        "application": "tempo-earth2-workshop",
        "begin_date": _iso_day(start).replace("-", ""),
        "end_date": _iso_day(end).replace("-", ""),
        "datum": datum,
        "station": station,
        "time_zone": "gmt",
        "units": units,
        "interval": interval,
        "format": "json",
    }
    return f"{NOAA_COOPS_URL}?{urlencode(query)}"


def read_noaa_coops(
    station: str,
    start: str | date | datetime,
    end: str | date | datetime,
    **kwargs,
) -> pd.DataFrame:
    """Read a bounded NOAA CO-OPS query into a tidy table."""
    url = noaa_coops_url(station, start, end, **kwargs)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "error" in payload:
        raise ValueError(payload["error"].get("message", str(payload["error"])))
    frame = pd.DataFrame(payload.get("data", payload.get("predictions", [])))
    if not frame.empty and "t" in frame:
        frame = frame.rename(columns={"t": "time_utc", "v": "value"})
        frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["station"] = station
    return frame
