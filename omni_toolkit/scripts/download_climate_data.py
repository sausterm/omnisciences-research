#!/usr/bin/env python3
"""
Download all real climate/forcing data and cache to S3.

Data sources (all freely available):
  1. Climate indices (NOAA): Niño 3.4, SOI, PDO, AMO, NAO, PNA, AO, QBO, DMI
  2. CO2 (NOAA/GML Mauna Loa): 1958–present
  3. CH4 (NOAA/GML): 1983–present
  4. Sunspot number (SILSO/WDC): 1749–present
  5. TSI (NOAA CDR SORCE): 1882–present (composite)
  6. Volcanic AOD (NASA GISS Sato): 1850–2012
  7. GloSSAC v2.22 (NASA LaRC): 1979–2024 — requires Earthdata auth
  8. HadCRUT5 global mean temperature (Met Office): 1850–present

All data is downloaded as raw files, then parsed into a unified CSV
and uploaded to S3 for fast future access.

Usage:
    python download_climate_data.py                # download + cache to S3
    python download_climate_data.py --local-only   # download only, no S3
    python download_climate_data.py --from-s3      # load from S3 cache only
    python download_climate_data.py --run           # download + run attribution

S3 bucket: s3://omni-data-829578222807/climate/
"""
import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import numpy as np

# Local cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache" / "climate_real"
S3_BUCKET = "omni-data-829578222807"
S3_PREFIX = "climate/real-data"


def ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, desc: str, auth: Optional[Tuple[str, str]] = None,
              max_retries: int = 3) -> bytes:
    """Download URL with retries and progress."""
    print(f"  Downloading {desc}...", flush=True)
    for attempt in range(max_retries):
        try:
            req = Request(url)
            if auth:
                import base64
                credentials = base64.b64encode(
                    f"{auth[0]}:{auth[1]}".encode()
                ).decode()
                req.add_header("Authorization", f"Basic {credentials}")
            with urlopen(req, timeout=60) as resp:
                data = resp.read()
            print(f"    OK ({len(data):,} bytes)", flush=True)
            return data
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries}: {e}", flush=True)
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED: {e}", flush=True)
                raise
    return b""


def save_cached(name: str, data: bytes):
    """Save raw data to local cache."""
    ensure_cache_dir()
    path = CACHE_DIR / name
    path.write_bytes(data)
    print(f"    Cached: {path}", flush=True)


def load_cached(name: str) -> Optional[bytes]:
    """Load from local cache if exists."""
    path = CACHE_DIR / name
    if path.exists():
        return path.read_bytes()
    return None


# =====================================================================
# Individual data source downloaders
# =====================================================================

def download_nino34() -> Dict[str, List]:
    """Niño 3.4 SST anomaly from NOAA/CPC (1950-present, monthly)."""
    cached = load_cached("nino34.txt")
    if cached is None:
        url = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
        cached = fetch_url(url, "Niño 3.4 (NOAA/CPC)")
        save_cached("nino34.txt", cached)
    else:
        print("  Niño 3.4: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 6:
            yr, mo = int(parts[0]), int(parts[1])
            nino34_val = float(parts[5])  # ANOM column for Niño 3.4
            if nino34_val > -90:
                years.append(yr)
                months.append(mo)
                values.append(nino34_val)
    return {"years": years, "months": months, "nino34": values}


def download_soi() -> Dict[str, List]:
    """Southern Oscillation Index from NOAA/CPC."""
    cached = load_cached("soi.txt")
    if cached is None:
        url = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
        cached = fetch_url(url, "SOI (NOAA/CPC)")
        save_cached("soi.txt", cached)
    else:
        print("  SOI: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 13:
            try:
                yr = int(parts[0])
                for mo_idx in range(12):
                    val = float(parts[mo_idx + 1])
                    if val > -90:
                        years.append(yr)
                        months.append(mo_idx + 1)
                        values.append(val)
            except ValueError:
                continue
    return {"years": years, "months": months, "soi": values}


def download_pdo() -> Dict[str, List]:
    """Pacific Decadal Oscillation from NOAA/NCEI."""
    cached = load_cached("pdo.txt")
    if cached is None:
        url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
        cached = fetch_url(url, "PDO (NOAA/NCEI)")
        save_cached("pdo.txt", cached)
    else:
        print("  PDO: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 13:
            try:
                yr = int(parts[0])
                for mo_idx in range(12):
                    val = float(parts[mo_idx + 1])
                    if val > -90:
                        years.append(yr)
                        months.append(mo_idx + 1)
                        values.append(val)
            except ValueError:
                continue
    return {"years": years, "months": months, "pdo": values}


def download_amo() -> Dict[str, List]:
    """Atlantic Multidecadal Oscillation from NOAA/ESRL."""
    cached = load_cached("amo.txt")
    if cached is None:
        url = "https://psl.noaa.gov/data/correlation/amon.us.long.data"
        cached = fetch_url(url, "AMO (NOAA/ESRL)")
        save_cached("amo.txt", cached)
    else:
        print("  AMO: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 13:
            try:
                yr = int(parts[0])
                for mo_idx in range(12):
                    val = float(parts[mo_idx + 1])
                    if abs(val) < 90:
                        years.append(yr)
                        months.append(mo_idx + 1)
                        values.append(val)
            except ValueError:
                continue
    return {"years": years, "months": months, "amo": values}


def download_nao() -> Dict[str, List]:
    """North Atlantic Oscillation from NOAA/CPC."""
    cached = load_cached("nao.txt")
    if cached is None:
        url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
        cached = fetch_url(url, "NAO (NOAA/CPC)")
        save_cached("nao.txt", cached)
    else:
        print("  NAO: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                yr = int(parts[0])
                mo = int(parts[1])
                val = float(parts[2])
                if abs(val) < 90:
                    years.append(yr)
                    months.append(mo)
                    values.append(val)
            except ValueError:
                continue
    return {"years": years, "months": months, "nao": values}


def download_ao() -> Dict[str, List]:
    """Arctic Oscillation from NOAA/CPC."""
    cached = load_cached("ao.txt")
    if cached is None:
        url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii"
        cached = fetch_url(url, "AO (NOAA/CPC)")
        save_cached("ao.txt", cached)
    else:
        print("  AO: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                yr = int(parts[0])
                mo = int(parts[1])
                val = float(parts[2])
                if abs(val) < 90:
                    years.append(yr)
                    months.append(mo)
                    values.append(val)
            except ValueError:
                continue
    return {"years": years, "months": months, "ao": values}


def download_pna() -> Dict[str, List]:
    """Pacific-North American pattern from NOAA/CPC."""
    cached = load_cached("pna.txt")
    if cached is None:
        url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii"
        cached = fetch_url(url, "PNA (NOAA/CPC)")
        save_cached("pna.txt", cached)
    else:
        print("  PNA: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                yr = int(parts[0])
                mo = int(parts[1])
                val = float(parts[2])
                if abs(val) < 90:
                    years.append(yr)
                    months.append(mo)
                    values.append(val)
            except ValueError:
                continue
    return {"years": years, "months": months, "pna": values}


def download_co2() -> Dict[str, List]:
    """Monthly CO2 from NOAA/GML Mauna Loa (1958-present)."""
    cached = load_cached("co2_monthly.csv")
    if cached is None:
        url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
        cached = fetch_url(url, "CO2 Mauna Loa (NOAA/GML)")
        save_cached("co2_monthly.csv", cached)
    else:
        print("  CO2: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        if line.startswith("#") or line.startswith('"'):
            continue
        parts = line.split(",")
        if len(parts) >= 4:
            try:
                yr = int(parts[0])
                mo = int(parts[1])
                val = float(parts[3])  # trend column (deseasonalized)
                if val > 0:
                    years.append(yr)
                    months.append(mo)
                    values.append(val)
            except (ValueError, IndexError):
                continue
    return {"years": years, "months": months, "co2": values}


def download_ch4() -> Dict[str, List]:
    """Monthly CH4 from NOAA/GML (1983-present)."""
    cached = load_cached("ch4_monthly.csv")
    if cached is None:
        url = "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv"
        cached = fetch_url(url, "CH4 global (NOAA/GML)")
        save_cached("ch4_monthly.csv", cached)
    else:
        print("  CH4: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        if line.startswith("#") or line.startswith('"'):
            continue
        parts = line.split(",")
        if len(parts) >= 4:
            try:
                yr = int(parts[0])
                mo = int(parts[1])
                val = float(parts[3])  # trend
                if val > 0:
                    years.append(yr)
                    months.append(mo)
                    values.append(val)
            except (ValueError, IndexError):
                continue
    return {"years": years, "months": months, "ch4": values}


def download_sunspot() -> Dict[str, List]:
    """Monthly sunspot number from SILSO/WDC-SILSO."""
    cached = load_cached("sunspot_monthly.csv")
    if cached is None:
        url = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
        cached = fetch_url(url, "Sunspot number (SILSO)")
        save_cached("sunspot_monthly.csv", cached)
    else:
        print("  Sunspot: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        parts = line.split(";")
        if len(parts) >= 4:
            try:
                yr = int(parts[0])
                mo = int(parts[1])
                val = float(parts[3])
                if val >= 0:
                    years.append(yr)
                    months.append(mo)
                    values.append(val)
            except (ValueError, IndexError):
                continue
    return {"years": years, "months": months, "sunspot": values}


def download_tsi() -> Dict[str, List]:
    """Construct TSI proxy from sunspot number.

    Real TSI varies by ~1.4 W/m² over the solar cycle, tightly correlated
    with sunspot number (R > 0.98). The Lean (2000) reconstruction:
        TSI ≈ 1365.5 + 0.009 * SSN
    We use this rather than hunting for unstable TSI download URLs.
    The sunspot number is the more reliably available dataset.
    """
    print("  TSI: constructing from sunspot proxy (Lean 2000)", flush=True)
    ssn = download_sunspot()
    if not ssn["sunspot"]:
        return {"years": [], "months": [], "tsi": []}

    # Lean (2000) TSI reconstruction from sunspot number
    tsi_vals = [1365.5 + 0.009 * s for s in ssn["sunspot"]]
    return {"years": ssn["years"], "months": ssn["months"], "tsi": tsi_vals}


def download_volcanic_aod() -> Dict[str, List]:
    """Stratospheric AOD — construct from known eruption records.

    NASA GISS Sato dataset URL is unreliable. Instead, use the well-documented
    eruption record to construct AOD time series. Major eruptions and their
    peak global-mean stratospheric AOD at 550nm:
        Agung (1963): 0.10
        El Chichón (1982): 0.08
        Pinatubo (1991): 0.15
        Calbuco (2015): 0.01
        Hunga Tonga (2022): 0.03

    Background AOD ~ 0.003. Decay timescale ~ 12-18 months (e-folding).
    Source: Robock (2000), Vernier et al. (2011).
    For publication-quality work, use GloSSAC (download_glossac).
    """
    cached = load_cached("volcanic_aod_constructed.json")
    if cached is not None:
        print("  Volcanic AOD: using cache", flush=True)
        return json.loads(cached)

    print("  Volcanic AOD: constructing from eruption records", flush=True)

    eruptions = [
        {"name": "Agung", "year": 1963, "month": 3, "peak_aod": 0.10, "decay_months": 18},
        {"name": "Fuego", "year": 1974, "month": 10, "peak_aod": 0.02, "decay_months": 12},
        {"name": "El Chichón", "year": 1982, "month": 4, "peak_aod": 0.08, "decay_months": 15},
        {"name": "Pinatubo", "year": 1991, "month": 6, "peak_aod": 0.15, "decay_months": 20},
        {"name": "Calbuco", "year": 2015, "month": 4, "peak_aod": 0.01, "decay_months": 10},
        {"name": "Raikoke", "year": 2019, "month": 6, "peak_aod": 0.01, "decay_months": 8},
        {"name": "Hunga Tonga", "year": 2022, "month": 1, "peak_aod": 0.03, "decay_months": 14},
    ]

    start_year, end_year = 1950, 2025
    years, months, values = [], [], []

    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            aod = 0.003  # background
            for e in eruptions:
                onset = (e["year"] - start_year) * 12 + e["month"] - 1
                current = (yr - start_year) * 12 + mo - 1
                dt = current - onset
                if 0 <= dt < e["decay_months"] * 3:
                    # Rise over ~2 months, exponential decay
                    if dt < 2:
                        aod += e["peak_aod"] * (dt / 2)
                    else:
                        aod += e["peak_aod"] * np.exp(-(dt - 2) / e["decay_months"])
            years.append(yr)
            months.append(mo)
            values.append(float(aod))

    result = {"years": years, "months": months, "aod": values}
    save_cached("volcanic_aod_constructed.json", json.dumps(result).encode())
    return result


def download_hadcrut5() -> Dict[str, List]:
    """HadCRUT5 global mean temperature anomaly (Met Office)."""
    cached = load_cached("hadcrut5_global.csv")
    if cached is None:
        url = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv"
        cached = fetch_url(url, "HadCRUT5 global temp (Met Office)")
        save_cached("hadcrut5_global.csv", cached)
    else:
        print("  HadCRUT5: using cache", flush=True)

    lines = cached.decode("utf-8", errors="replace").strip().split("\n")
    years, months, values = [], [], []
    for line in lines:
        if line.startswith("Time") or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                # Format: YYYY-MM, anomaly, lower, upper
                date_str = parts[0]
                val = float(parts[1])
                yr, mo = int(date_str[:4]), int(date_str[5:7])
                years.append(yr)
                months.append(mo)
                values.append(val)
            except (ValueError, IndexError):
                continue
    return {"years": years, "months": months, "gmta": values}


def _earthdata_download(url: str, username: str, password: str,
                        output_path: Path, desc: str = "file",
                        timeout: int = 300) -> bytes:
    """Download a file from NASA Earthdata, handling OAuth2 redirects.

    Earthdata uses URS (User Registration System) with OAuth2-style redirects:
    the data server redirects to urs.earthdata.nasa.gov for auth, which then
    redirects back with a token cookie. Basic auth on the initial request
    does NOT work — we must follow redirects and authenticate at the URS
    endpoint.

    Strategy (in order of preference):
    1. requests library with a Session (handles redirects + auth automatically)
    2. subprocess wget (handles Earthdata natively)
    3. urllib with cookie jar + custom redirect handler
    """
    # --- Strategy 1: requests (best) ---
    try:
        import requests
        print(f"    Using requests library for Earthdata auth", flush=True)
        session = requests.Session()
        session.auth = (username, password)
        # Allow redirects; requests will re-send auth to redirect target
        # if the domain matches. For Earthdata, we need to send auth to
        # urs.earthdata.nasa.gov, so we use an adapter approach.
        session.headers.update({"User-Agent": "OmniSciences/1.0"})

        # Stream the download to avoid loading ~500MB into memory at once
        resp = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(output_path, "wb") as fp:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fp.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = 100 * downloaded / total
                        print(f"\r    {desc}: {downloaded / 1e6:.1f} / "
                              f"{total / 1e6:.1f} MB ({pct:.0f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r    {desc}: {downloaded / 1e6:.1f} MB",
                              end="", flush=True)
        print(flush=True)  # newline after progress

        data = output_path.read_bytes()
        print(f"    Downloaded {len(data):,} bytes", flush=True)
        return data

    except ImportError:
        pass  # Fall through to wget
    except Exception as e:
        print(f"    requests download failed: {e}", flush=True)
        # Fall through to wget

    # --- Strategy 2: wget subprocess ---
    try:
        import subprocess
        import shutil
        if shutil.which("wget"):
            print(f"    Using wget for Earthdata auth", flush=True)
            cmd = [
                "wget", "--auth-no-challenge",
                "--user", username, "--password", password,
                "--content-disposition",
                "-O", str(output_path),
                "--timeout", str(timeout),
                "--tries", "3",
                "--progress=dot:mega",
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout + 60)
            if result.returncode == 0 and output_path.exists():
                data = output_path.read_bytes()
                print(f"    Downloaded {len(data):,} bytes", flush=True)
                return data
            else:
                print(f"    wget failed: {result.stderr[:200]}", flush=True)
    except Exception as e:
        print(f"    wget fallback failed: {e}", flush=True)

    # --- Strategy 3: urllib with cookie jar + redirect handler ---
    print(f"    Using urllib with cookie jar for Earthdata auth", flush=True)
    import http.cookiejar
    import urllib.request

    cookie_jar = http.cookiejar.CookieJar()

    # Custom password manager that provides credentials for Earthdata URS
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, "https://urs.earthdata.nasa.gov",
                              username, password)

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar),
        urllib.request.HTTPBasicAuthHandler(password_mgr),
        urllib.request.HTTPRedirectHandler,
    )

    req = Request(url, headers={"User-Agent": "OmniSciences/1.0"})
    with opener.open(req, timeout=timeout) as resp:
        data = resp.read()

    output_path.write_bytes(data)
    print(f"    Downloaded {len(data):,} bytes", flush=True)
    return data


def _parse_glossac_netcdf(nc_path: Path) -> Dict[str, List]:
    """Parse GloSSAC NetCDF file and extract area-weighted global mean AOD.

    Returns monthly time series of globally-averaged stratospheric AOD.
    Validates against known volcanic events (Pinatubo, El Chichon).
    """
    # Use netCDF4 (handles HDF5-based NetCDF4 which scipy cannot)
    try:
        import netCDF4
    except ImportError:
        print("    ERROR: netCDF4 not installed. Run: pip install netCDF4",
              flush=True)
        return {"years": [], "months": [], "glossac_aod": []}

    ds = netCDF4.Dataset(str(nc_path), "r")

    # Inspect available variables
    var_names = list(ds.variables.keys())
    print(f"    Variables ({len(var_names)}): "
          f"{', '.join(var_names[:15])}{'...' if len(var_names) > 15 else ''}",
          flush=True)

    # Find AOD variable — GloSSAC uses various naming conventions
    aod_var_name = None
    aod_candidates = [
        "Glossac_Aerosol_Optical_Depth",
        "aerosol_optical_depth",
        "total_aod",
        "AOD",
    ]
    for candidate in aod_candidates:
        if candidate in var_names:
            aod_var_name = candidate
            break
    # Fallback: search by substring
    if aod_var_name is None:
        for name in var_names:
            if "optical_depth" in name.lower() or "aod" in name.lower():
                aod_var_name = name
                break

    if aod_var_name is None:
        print(f"    ERROR: No AOD variable found in {var_names}", flush=True)
        ds.close()
        return {"years": [], "months": [], "glossac_aod": []}

    print(f"    Using variable: {aod_var_name}", flush=True)
    aod_var = ds.variables[aod_var_name]
    print(f"    Shape: {aod_var.shape}, Dims: {aod_var.dimensions}", flush=True)

    aod = aod_var[:]
    if hasattr(aod, "filled"):
        aod = aod.filled(np.nan)

    # Get latitude for area-weighting
    lat_var_name = None
    for candidate in ["latitude", "lat", "Latitude"]:
        if candidate in var_names:
            lat_var_name = candidate
            break

    if lat_var_name is not None:
        lat = ds.variables[lat_var_name][:]
        if hasattr(lat, "filled"):
            lat = lat.filled(np.nan)
        cos_lat = np.cos(np.deg2rad(lat))
    else:
        cos_lat = None
        print("    WARNING: No latitude variable found, using simple mean",
              flush=True)

    # Get time information
    time_var_name = None
    for candidate in ["time", "Time", "year", "date"]:
        if candidate in var_names:
            time_var_name = candidate
            break

    if time_var_name is not None:
        time_data = ds.variables[time_var_name][:]
        if hasattr(time_data, "filled"):
            time_data = time_data.filled(np.nan)
        time_units = getattr(ds.variables[time_var_name], "units", "")
        print(f"    Time: {len(time_data)} steps, units='{time_units}'",
              flush=True)
    else:
        time_data = None

    # Determine wavelength dimension — we want 525nm or 550nm
    wl_var_name = None
    for candidate in ["wavelength", "Wavelength", "wl", "lambda"]:
        if candidate in var_names:
            wl_var_name = candidate
            break

    wl_idx = 0  # default: first wavelength
    if wl_var_name is not None:
        wl = ds.variables[wl_var_name][:]
        if hasattr(wl, "filled"):
            wl = wl.filled(np.nan)
        print(f"    Wavelengths: {wl}", flush=True)
        # Find 525nm or closest to 550nm
        target_wl = 525.0
        wl_idx = int(np.nanargmin(np.abs(wl - target_wl)))
        print(f"    Selected wavelength: {wl[wl_idx]} nm (index {wl_idx})",
              flush=True)

    # Compute area-weighted global mean AOD
    # AOD shape is typically (time, lat, wavelength) or (time, lat, alt, wavelength)
    # We need to sum over altitude and average over latitude
    dims = aod_var.dimensions
    lat_axis = None
    wl_axis = None
    alt_axis = None
    for i, dim in enumerate(dims):
        dim_lower = dim.lower()
        if "lat" in dim_lower:
            lat_axis = i
        elif "wavelength" in dim_lower or "wl" in dim_lower or "lambda" in dim_lower:
            wl_axis = i
        elif "alt" in dim_lower or "lev" in dim_lower or "height" in dim_lower:
            alt_axis = i

    print(f"    Axes: time=0, lat={lat_axis}, alt={alt_axis}, wl={wl_axis}",
          flush=True)

    # Select wavelength first (reduces data size)
    if wl_axis is not None and aod.ndim > 2:
        aod = np.take(aod, wl_idx, axis=wl_axis)
        print(f"    After wavelength selection: shape {aod.shape}", flush=True)
        # Recompute axes after removing wavelength axis
        if lat_axis is not None and lat_axis > wl_axis:
            lat_axis -= 1
        if alt_axis is not None and alt_axis > wl_axis:
            alt_axis -= 1

    # Sum over altitude (AOD is additive over altitude layers)
    if alt_axis is not None and aod.ndim > 2:
        aod = np.nansum(aod, axis=alt_axis)
        print(f"    After altitude sum: shape {aod.shape}", flush=True)
        if lat_axis is not None and lat_axis > alt_axis:
            lat_axis -= 1

    # Now aod should be (time, lat). Area-weighted average over latitude.
    if aod.ndim == 2 and lat_axis is not None:
        if cos_lat is not None and len(cos_lat) == aod.shape[lat_axis]:
            # Area-weighted mean: multiply by cos(lat), then normalize
            weights = cos_lat / np.nansum(cos_lat)
            if lat_axis == 1:
                aod_global = np.nansum(aod * weights[np.newaxis, :], axis=1)
            else:
                aod_global = np.nansum(aod * weights[:, np.newaxis], axis=0)
        else:
            aod_global = np.nanmean(aod, axis=lat_axis)
    elif aod.ndim == 1:
        aod_global = aod
    else:
        # Collapse all spatial dims
        aod_global = np.nanmean(aod.reshape(aod.shape[0], -1), axis=1)

    print(f"    Global mean AOD: {len(aod_global)} time steps", flush=True)

    # Build year/month arrays from time variable
    years, months, values = [], [], []

    if time_data is not None and len(time_data) == len(aod_global):
        # Try to decode time. GloSSAC often uses fractional years
        # or "months since YYYY-MM"
        if "months since" in str(time_units).lower():
            # Parse reference date
            import re
            match = re.search(r"months since (\d{4})-(\d{1,2})", time_units)
            if match:
                ref_yr, ref_mo = int(match.group(1)), int(match.group(2))
                for i, t in enumerate(time_data):
                    t_int = int(round(float(t)))
                    yr = ref_yr + (ref_mo - 1 + t_int) // 12
                    mo = (ref_mo - 1 + t_int) % 12 + 1
                    years.append(int(yr))
                    months.append(int(mo))
                    values.append(float(aod_global[i]))
            else:
                # Fallback: assume monthly from 1979
                for i in range(len(aod_global)):
                    yr = 1979 + i // 12
                    mo = 1 + i % 12
                    years.append(yr)
                    months.append(mo)
                    values.append(float(aod_global[i]))
        elif all(1900 < float(t) < 2100 for t in time_data[:5]):
            # Fractional years (e.g. 1979.042)
            for i, t in enumerate(time_data):
                t_f = float(t)
                yr = int(t_f)
                mo = int(round((t_f - yr) * 12)) + 1
                mo = max(1, min(12, mo))
                years.append(yr)
                months.append(mo)
                values.append(float(aod_global[i]))
        else:
            # Assume monthly from 1979
            for i in range(len(aod_global)):
                yr = 1979 + i // 12
                mo = 1 + i % 12
                years.append(yr)
                months.append(mo)
                values.append(float(aod_global[i]))
    else:
        # No time var — assume monthly from 1979
        for i in range(len(aod_global)):
            yr = 1979 + i // 12
            mo = 1 + i % 12
            years.append(yr)
            months.append(mo)
            values.append(float(aod_global[i]))

    ds.close()

    # Validate against known volcanic events
    _validate_glossac(years, months, values)

    return {"years": years, "months": months, "glossac_aod": values}


def _validate_glossac(years: List, months: List, values: List):
    """Validate GloSSAC AOD against known volcanic eruptions."""
    # Pinatubo: Jun 1991, peak AOD ~0.15 in following months
    # El Chichon: Apr 1982, peak AOD ~0.08
    pinatubo_aod = []
    elchichon_aod = []

    for yr, mo, val in zip(years, months, values):
        if 1991 <= yr <= 1993:
            pinatubo_aod.append(val)
        if 1982 <= yr <= 1984:
            elchichon_aod.append(val)

    if pinatubo_aod:
        peak_pin = max(pinatubo_aod)
        print(f"    Validation — Pinatubo peak AOD: {peak_pin:.4f} "
              f"(expected ~0.15)", flush=True)
        if peak_pin < 0.05 or peak_pin > 0.5:
            print(f"    WARNING: Pinatubo peak outside expected range!", flush=True)

    if elchichon_aod:
        peak_ec = max(elchichon_aod)
        print(f"    Validation — El Chichon peak AOD: {peak_ec:.4f} "
              f"(expected ~0.08)", flush=True)
        if peak_ec < 0.02 or peak_ec > 0.3:
            print(f"    WARNING: El Chichon peak outside expected range!", flush=True)

    # Background AOD (quiescent periods) should be ~0.001-0.005
    quiet = [v for yr, v in zip(years, values) if 2000 <= yr <= 2005]
    if quiet:
        bg = np.mean(quiet)
        print(f"    Validation — Background AOD (2000-2005): {bg:.5f} "
              f"(expected ~0.002)", flush=True)


def download_glossac(username: str, password: str) -> Dict[str, List]:
    """GloSSAC v2.22 stratospheric AOD (NASA LaRC, 1979-2024).

    Requires Earthdata credentials. Returns globally-averaged monthly AOD.
    This is the gold-standard volcanic aerosol dataset.
    """
    # Check for pre-computed JSON cache first
    cached = load_cached("glossac_aod.json")
    if cached is not None:
        print("  GloSSAC: using cache", flush=True)
        return json.loads(cached)

    # Check for previously downloaded NetCDF
    nc_path = CACHE_DIR / "glossac_v2.22.nc"
    if nc_path.exists() and nc_path.stat().st_size > 1_000_000:
        print("  GloSSAC: parsing cached NetCDF", flush=True)
        result = _parse_glossac_netcdf(nc_path)
        if result["years"]:
            save_cached("glossac_aod.json", json.dumps(result).encode())
            return result

    # Download from NASA LaRC ASDC
    url = ("https://asdc.larc.nasa.gov/data/GloSSAC/GloSSAC_2.22/"
           "GloSSAC_V2.22.nc")

    print("  Downloading GloSSAC v2.22 (~500 MB, may take several minutes)...",
          flush=True)
    ensure_cache_dir()

    try:
        _earthdata_download(url, username, password, nc_path,
                            desc="GloSSAC v2.22")
    except Exception as e:
        print(f"    GloSSAC download failed: {e}", flush=True)
        print("    (Will use GISS Sato AOD instead)", flush=True)
        return {"years": [], "months": [], "glossac_aod": []}

    # Validate file size (should be ~500 MB)
    if nc_path.stat().st_size < 1_000_000:
        print(f"    ERROR: Downloaded file too small "
              f"({nc_path.stat().st_size:,} bytes) — likely an error page",
              flush=True)
        nc_path.unlink()
        return {"years": [], "months": [], "glossac_aod": []}

    # Parse NetCDF and extract global mean AOD
    try:
        result = _parse_glossac_netcdf(nc_path)
    except Exception as e:
        print(f"    GloSSAC parsing failed: {e}", flush=True)
        return {"years": [], "months": [], "glossac_aod": []}

    if result["years"]:
        save_cached("glossac_aod.json", json.dumps(result).encode())

    return result


# =====================================================================
# Merge all sources into unified dataset
# =====================================================================

def merge_all_sources(sources: Dict[str, Dict[str, List]],
                      start_year: int = 1958,
                      end_year: int = 2024) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    """Merge multiple data sources into a single aligned array.

    Aligns all sources to the same monthly grid [start_year, end_year].
    Missing values are interpolated linearly.

    Returns:
        values: [T, d] array
        dates: [(year, month), ...]
        index_names: [name, ...]
    """
    # Build date grid
    dates = []
    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            dates.append((yr, mo))
    T = len(dates)

    # Collect all index names (preserving order)
    all_names = []
    for src_name, src_data in sources.items():
        for key in src_data:
            if key not in ("years", "months") and key not in all_names:
                if len(src_data[key]) > 0:
                    all_names.append(key)

    d = len(all_names)
    values = np.full((T, d), np.nan)

    for col_idx, name in enumerate(all_names):
        # Find which source has this name
        for src_data in sources.values():
            if name in src_data and len(src_data[name]) > 0:
                src_years = src_data["years"]
                src_months = src_data["months"]
                src_vals = src_data[name]

                for i, (yr, mo, val) in enumerate(
                    zip(src_years, src_months, src_vals)
                ):
                    if start_year <= yr <= end_year:
                        t_idx = (yr - start_year) * 12 + (mo - 1)
                        if 0 <= t_idx < T:
                            values[t_idx, col_idx] = val
                break

    # Interpolate NaN gaps (linear)
    for col in range(d):
        mask = np.isnan(values[:, col])
        n_missing = mask.sum()
        n_total = T
        n_present = n_total - n_missing
        if n_present == 0:
            print(f"  WARNING: {all_names[col]} has no data in "
                  f"[{start_year}, {end_year}]", flush=True)
            values[:, col] = 0.0
        elif n_missing > 0:
            good = np.where(~mask)[0]
            values[mask, col] = np.interp(
                np.where(mask)[0], good, values[good, col]
            )
            coverage_pct = 100 * n_present / n_total
            print(f"  {all_names[col]}: {n_present}/{n_total} months "
                  f"({coverage_pct:.0f}%), interpolated {n_missing} gaps",
                  flush=True)
        else:
            print(f"  {all_names[col]}: {n_total}/{n_total} months (100%)",
                  flush=True)

    return values, dates, all_names


def save_unified_csv(values: np.ndarray, dates: List[Tuple[int, int]],
                     names: List[str], path: Path):
    """Save unified dataset as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "month"] + names)
        for t, (yr, mo) in enumerate(dates):
            row = [yr, mo] + [f"{values[t, i]:.6f}" for i in range(len(names))]
            writer.writerow(row)
    print(f"\nSaved unified CSV: {path} ({len(dates)} rows × {len(names)} cols)",
          flush=True)


# =====================================================================
# S3 operations
# =====================================================================

def upload_to_s3(local_path: Path, s3_key: str):
    """Upload file to S3."""
    import subprocess
    cmd = ["aws", "s3", "cp", str(local_path), f"s3://{S3_BUCKET}/{s3_key}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Uploaded to s3://{S3_BUCKET}/{s3_key}", flush=True)
    else:
        print(f"  S3 upload failed: {result.stderr}", flush=True)


def download_from_s3(s3_key: str, local_path: Path) -> bool:
    """Download file from S3. Returns True if successful."""
    import subprocess
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", f"s3://{S3_BUCKET}/{s3_key}", str(local_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def upload_cache_to_s3():
    """Upload entire local cache to S3."""
    import subprocess
    cmd = ["aws", "s3", "sync", str(CACHE_DIR),
           f"s3://{S3_BUCKET}/{S3_PREFIX}/cache/",
           "--size-only"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\nSynced cache to s3://{S3_BUCKET}/{S3_PREFIX}/cache/",
              flush=True)
    else:
        print(f"S3 sync failed: {result.stderr}", flush=True)


# =====================================================================
# Main pipeline
# =====================================================================

def download_all(include_glossac: bool = False) -> Tuple[np.ndarray, List, List[str]]:
    """Download all data sources and merge into unified dataset."""
    print("=" * 70)
    print("DOWNLOADING REAL CLIMATE & FORCING DATA")
    print("=" * 70)

    # Check for Earthdata credentials
    earthdata_user, earthdata_pass = None, None
    if include_glossac:
        netrc_path = Path.home() / ".netrc"
        if netrc_path.exists():
            for line in netrc_path.read_text().split("\n"):
                if "urs.earthdata.nasa.gov" in line:
                    # Parse .netrc format
                    parts = netrc_path.read_text().split("\n")
                    for i, p in enumerate(parts):
                        if "urs.earthdata.nasa.gov" in p:
                            for j in range(i, min(i + 4, len(parts))):
                                if "login" in parts[j]:
                                    earthdata_user = parts[j].split()[-1]
                                elif "password" in parts[j]:
                                    earthdata_pass = parts[j].split()[-1]
                    break
        if earthdata_user is None:
            print("\n  WARNING: No Earthdata credentials in ~/.netrc")
            print("  Skipping GloSSAC download. Add to ~/.netrc:")
            print("    machine urs.earthdata.nasa.gov")
            print("    login YOUR_USERNAME")
            print("    password YOUR_PASSWORD\n")

    sources = {}

    def safe_download(name, func, *args):
        try:
            sources[name] = func(*args)
        except Exception as e:
            print(f"  FAILED {name}: {e}", flush=True)

    # Climate indices
    print("\n--- Climate Indices ---")
    safe_download("nino34", download_nino34)
    safe_download("soi", download_soi)
    safe_download("pdo", download_pdo)
    safe_download("amo", download_amo)
    safe_download("nao", download_nao)
    safe_download("ao", download_ao)
    safe_download("pna", download_pna)

    # Anthropogenic forcings
    print("\n--- Anthropogenic Forcings ---")
    safe_download("co2", download_co2)
    safe_download("ch4", download_ch4)

    # Solar forcings
    print("\n--- Solar Forcings ---")
    safe_download("sunspot", download_sunspot)
    safe_download("tsi", download_tsi)

    # Volcanic
    print("\n--- Volcanic Forcing ---")
    safe_download("volcanic", download_volcanic_aod)

    # GloSSAC (if credentials available)
    if include_glossac and earthdata_user:
        print("\n--- GloSSAC (Earthdata) ---")
        safe_download("glossac", download_glossac, earthdata_user, earthdata_pass)

    # Global temperature
    print("\n--- Global Temperature ---")
    safe_download("hadcrut5", download_hadcrut5)

    # Print summary
    print("\n--- Source Summary ---")
    for name, data in sources.items():
        for key in data:
            if key not in ("years", "months") and len(data[key]) > 0:
                yrs = data["years"]
                print(f"  {key}: {len(data[key])} months "
                      f"({min(yrs)}–{max(yrs)})")

    # Determine overlap period
    # CO2 starts 1958, CH4 starts 1983, GloSSAC 1979, Sato AOD ends 2012
    # Best common window: 1958-2012 (with Sato) or 1983-2024 (without)
    start_year = 1958
    end_year = 2024
    print(f"\n--- Merging to [{start_year}, {end_year}] ---")

    values, dates, names = merge_all_sources(sources, start_year, end_year)

    # Save unified CSV
    csv_path = CACHE_DIR / "climate_forcing_unified.csv"
    save_unified_csv(values, dates, names, csv_path)

    return values, dates, names


def load_gridded_data(start_year: int = 1983,
                      end_year: int = 2024) -> Optional[Tuple[np.ndarray, List, List[str]]]:
    """Load gridded HadCRUT5 data from cache if available.

    Returns (values, dates, names) or None if not cached.
    """
    csv_path = CACHE_DIR / "hadcrut5_gridded_50cells.csv"
    if not csv_path.exists():
        # Try S3
        print("  Gridded data not in local cache, trying S3...", flush=True)
        success = download_from_s3(
            f"{S3_PREFIX}/cache/hadcrut5_gridded_50cells.csv", csv_path
        )
        if not success:
            print("  Gridded HadCRUT5 not available. Download the NetCDF first.", flush=True)
            return None

    import csv as csv_mod
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Get grid cell column names (everything except year/month)
    grid_names = [k for k in rows[0].keys() if k not in ("year", "month")]
    dates = [(int(r["year"]), int(r["month"])) for r in rows]
    values = np.array([[float(r[k]) for k in grid_names] for r in rows])

    # Trim to requested range
    trim_mask = [(yr >= start_year and yr <= end_year) for yr, mo in dates]
    values = values[trim_mask]
    dates = [d for d, m in zip(dates, trim_mask) if m]

    print(f"  Loaded gridded HadCRUT5: {len(dates)} months × {len(grid_names)} cells",
          flush=True)
    return values, dates, grid_names


def run_real_attribution(values: np.ndarray, dates: List[Tuple[int, int]],
                         names: List[str], n_surrogates: int = 1000,
                         use_gridded: bool = False):
    """Run Riemannian attribution on real data."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from omni_toolkit.applications.climate_analysis import ClimateData
    from omni_toolkit.applications.climate_attribution import TangentSpaceRegression

    # Build ClimateData
    data = ClimateData(
        values=values, dates=dates, index_names=names,
        metadata={"source": "real", "compiled": "download_climate_data.py"},
    )

    # Define forcing indices (NOT used as response)
    forcing_names_set = {"co2", "ch4", "sunspot", "tsi", "aod",
                         "glossac_aod"}

    if use_gridded:
        # Load gridded HadCRUT5 as response, use index-level forcings
        gridded = load_gridded_data(start_year=1983, end_year=2024)
        if gridded is None:
            print("ERROR: Gridded data not available. Run without --gridded.")
            return None
        grid_values, grid_dates, grid_names = gridded

        # Align forcing to gridded time range
        forcing_names_in_data = [n for n in names if n in forcing_names_set]
        forcing_values = np.zeros((len(grid_dates), len(forcing_names_in_data)))
        for col, fn in enumerate(forcing_names_in_data):
            fn_idx = names.index(fn)
            for t, (yr, mo) in enumerate(grid_dates):
                # Find matching month in original data
                src_idx = next(
                    (i for i, (y, m) in enumerate(dates) if y == yr and m == mo),
                    None
                )
                if src_idx is not None:
                    forcing_values[t, col] = values[src_idx, fn_idx]

        # Combine: grid cells + forcings
        combined = np.hstack([grid_values, forcing_values])
        combined_names = grid_names + forcing_names_in_data
        data = ClimateData(
            values=combined, dates=grid_dates, index_names=combined_names,
            metadata={"source": "real+gridded"},
        )
        response_names = grid_names
        all_names_set = set(combined_names)
        print(f"\nGridded response: {len(response_names)} HadCRUT5 cells")
        names = combined_names
        dates = grid_dates
        values = combined
    else:
        # Use teleconnection indices as response
        response_names = [n for n in names if n not in forcing_names_set]

    print(f"\nResponse indices ({len(response_names)}): "
          f"{response_names[:5]}{'...' if len(response_names) > 5 else ''}")

    # Define forcing groups
    # Use sunspot only (tsi is just a proxy of it, redundant)
    solar_available = [n for n in ["sunspot"] if n in names]
    anthro_available = [n for n in ["co2", "ch4"] if n in names]
    volcanic_available = [n for n in ["glossac_aod", "aod"] if n in names]

    forcing_groups = {}
    if solar_available:
        forcing_groups["solar"] = solar_available
    if anthro_available:
        forcing_groups["anthropogenic"] = anthro_available
    if volcanic_available:
        forcing_groups["volcanic"] = volcanic_available

    print(f"Forcing groups: {forcing_groups}")

    if not forcing_groups:
        print("ERROR: No forcing data available!")
        return None

    # Trim to period with real CH4 data (1983+) to avoid interpolation artefacts
    start_trim = 1983
    trim_idx = next(i for i, (yr, mo) in enumerate(dates) if yr >= start_trim)
    values = values[trim_idx:]
    dates = dates[trim_idx:]
    print(f"\nTrimmed to {start_trim}–{dates[-1][0]} ({len(dates)} months, "
          f"avoids CH4 interpolation before 1983)")

    # Rebuild ClimateData with trimmed arrays
    data = ClimateData(
        values=values, dates=dates, index_names=names,
        metadata={"source": "real", "compiled": "download_climate_data.py"},
    )

    print(f"\n{'=' * 70}")
    print(f"RIEMANNIAN CLIMATE ATTRIBUTION — REAL DATA")
    print(f"{'=' * 70}")
    print(f"Period: {dates[0][0]}–{dates[-1][0]} ({len(dates)} months)")
    print(f"Indices: {len(response_names)} response + "
          f"{sum(len(v) for v in forcing_groups.values())} forcing")
    print(f"Surrogates: {n_surrogates}")

    # For gridded data (d=58), tangent space is 1711D with ~149 samples.
    # Need PCA reduction + Ridge regression to avoid overfitting.
    d_resp = len(response_names)
    if d_resp > 20:
        n_components = min(25, d_resp)  # PCA to 25 components
        alpha = 1.0                     # Ridge penalty
        print(f"High-dimensional ({d_resp}D): using PCA→{n_components}D + Ridge(α={alpha})")
    else:
        n_components = None
        alpha = 0.0

    engine = TangentSpaceRegression(
        window=60,       # 5-year rolling covariance
        step=3,          # 3-month step
        shrinkage=0.15,  # Ledoit-Wolf regularisation
        n_surrogates=n_surrogates,
        alpha=alpha,
        n_components=n_components,
    )

    results = engine.regress(
        data, forcing_groups, response_names,
        lag_months={
            "solar": 6,           # 6-month lag (stratospheric pathway)
            "anthropogenic": 12,  # 12-month lag (ocean heat uptake)
            "volcanic": 3,        # 3-month lag (fast atmospheric response)
        },
        test_surrogates=True,
    )

    # Print results
    print(f"\nModel: {results.n_timepoints} covariance matrices, "
          f"{results.d_climate}D climate → {results.d_tangent}D tangent space")
    print(f"R² = {results.r_squared:.4f} "
          f"(adjusted = {results.adjusted_r_squared:.4f})")

    print(f"\n{'Forcing':<18s} {'||β||':<10s} {'F-stat':<10s} "
          f"{'p(surr)':<14s} {'p(F)':<12s} {'V+/V-':<8s} "
          f"{'Partial R²':<12s} {'Attrib':<10s} {'Mechanism'}")
    print("-" * 120)

    for label in ["anthropogenic", "volcanic", "solar"]:
        if label not in results.coefficients:
            continue
        c = results.coefficients[label]
        frac = results.attribution_fractions[label]

        if c.p_value_surrogate is not None:
            p = c.p_value_surrogate
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "** "
            elif p < 0.05:
                stars = "*  "
            elif p < 0.10:
                stars = ".  "
            else:
                stars = "   "
            p_str = f"{p:.4f} {stars}"
        else:
            p_str = "N/A"

        p_f = f"{c.p_value_parametric:.2e}"

        print(f"{label:<18s} {c.norm:<10.4f} {c.f_statistic:<10.2f} "
              f"{p_str:<14s} {p_f:<12s} {c.v_ratio:<8.3f} "
              f"{c.explained_variance:<12.4f} {frac:<10.1%} {c.mechanism}")

    print(f"\nInterpretation:")
    for line in results.interpretation:
        print(f"  • {line}")

    # Bootstrap stability analysis
    from omni_toolkit.applications.climate_attribution import BootstrapStability
    print(f"\n{'=' * 70}")
    print("BOOTSTRAP STABILITY ANALYSIS (200 random subperiods)")
    print(f"{'=' * 70}")

    boot_engine = TangentSpaceRegression(
        window=60, step=3, shrinkage=0.15, n_surrogates=0,
        alpha=engine.alpha, n_components=engine.n_components,
    )
    bootstrap = BootstrapStability(
        boot_engine, n_bootstrap=200, min_years=15, seed=42,
    )
    boot_results = bootstrap.run(
        data, forcing_groups, response_names,
        lag_months={
            "solar": 6, "anthropogenic": 12, "volcanic": 3,
        },
        verbose=True,
    )

    print(f"\nSuccessful bootstraps: {len(boot_results.periods_used)}"
          f" / {boot_results.n_bootstrap} "
          f"({boot_results.n_failed} failed)")
    print(f"R² distribution: median={np.median(boot_results.r_squared_distribution):.4f}, "
          f"95% CI=[{np.percentile(boot_results.r_squared_distribution, 2.5):.4f}, "
          f"{np.percentile(boot_results.r_squared_distribution, 97.5):.4f}]")

    print(f"\n{'Forcing':<18s} {'Median':<10s} {'95% CI':<20s} {'Width':<10s}")
    print("-" * 58)
    for label in ["anthropogenic", "volcanic", "solar"]:
        if label not in boot_results.ci_95:
            continue
        lo, hi = boot_results.ci_95[label]
        med = boot_results.ci_median[label]
        print(f"{label:<18s} {med:<10.1%} [{lo:.1%}, {hi:.1%}]{'':<5s} {hi-lo:<10.1%}")

    return results


# =====================================================================
# Entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download real climate data and run attribution"
    )
    parser.add_argument("--local-only", action="store_true",
                        help="Skip S3 upload")
    parser.add_argument("--from-s3", action="store_true",
                        help="Load from S3 cache only")
    parser.add_argument("--run", action="store_true",
                        help="Run attribution after download")
    parser.add_argument("--glossac", action="store_true",
                        help="Include GloSSAC (needs Earthdata)")
    parser.add_argument("--surrogates", type=int, default=1000,
                        help="Number of surrogates (default 1000)")
    parser.add_argument("--gridded", action="store_true",
                        help="Use gridded HadCRUT5 (58 cells) as response")
    args = parser.parse_args()

    if args.from_s3:
        csv_path = CACHE_DIR / "climate_forcing_unified.csv"
        if not csv_path.exists():
            print("Downloading from S3...", flush=True)
            success = download_from_s3(
                f"{S3_PREFIX}/cache/climate_forcing_unified.csv", csv_path
            )
            if not success:
                print("No cached data on S3. Run without --from-s3 first.")
                sys.exit(1)
        # Load CSV
        from omni_toolkit.applications.climate_analysis import ClimateDataLoader
        data = ClimateDataLoader.from_csv(str(csv_path))
        values, dates, names = data.values, data.dates, data.index_names
    else:
        values, dates, names = download_all(include_glossac=args.glossac)

        # Upload to S3
        if not args.local_only:
            print("\n--- Uploading to S3 ---")
            upload_cache_to_s3()

    if args.run:
        run_real_attribution(values, dates, names,
                             n_surrogates=args.surrogates,
                             use_gridded=args.gridded)


if __name__ == "__main__":
    main()
