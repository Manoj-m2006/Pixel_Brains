from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection,
    MimeType, CRS, BBox, MosaickingOrder,
)
import datetime, math
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ── Credentials ───────────────────────────────────────────────────────────────
# IMPORTANT: Set these as environment variables or in a .env file
# Never commit credentials to GitHub!

config = SHConfig()
config.sh_client_id     = os.getenv('SH_CLIENT_ID', 'YOUR_CLIENT_ID_HERE')
config.sh_client_secret = os.getenv('SH_CLIENT_SECRET', 'YOUR_CLIENT_SECRET_HERE')

# ── Evalscripts ───────────────────────────────────────────────────────────────
_OPTICAL_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B04","B03","B02","SCL"] }],
        output: { bands: 3, sampleType: "AUTO" },
        mosaicking: "ORBIT"
    };
}
function evaluatePixel(samples) {
    // Safety: no data available
    if (!samples || samples.length === 0) {
        return [0, 0, 0];
    }
    // SCL cloud classes: 3=shadow, 8=medium, 9=high, 10=cirrus, 1=saturated, 0=no_data
    var cloudClasses = [0, 1, 3, 8, 9, 10, 11];
    var clearSamples = [];
    
    // Collect all clear samples
    for (var i = 0; i < samples.length; i++) {
        var s = samples[i];
        if (s && s.B04 !== undefined && cloudClasses.indexOf(s.SCL) === -1) {
            clearSamples.push(s);
        }
    }
    
    // If we have clear samples, use the most recent one
    if (clearSamples.length > 0) {
        var s = clearSamples[clearSamples.length - 1];
        return [3.2 * s.B04, 3.2 * s.B03, 3.2 * s.B02];
    }
    
    // No clear samples - use ANY available sample (even cloudy > nothing)
    for (var i = samples.length - 1; i >= 0; i--) {
        var s = samples[i];
        if (s && s.B04 !== undefined) {
            return [3.2 * s.B04, 3.2 * s.B03, 3.2 * s.B02];
        }
    }
    
    // Truly no data
    return [0, 0, 0];
}
"""

# SAR Sentinel-1 — VV backscatter, normalized to 0-255 display range
# dB range: -25 dB (water/smooth) to 0 dB (urban/rough).  Mapped linearly to 0-255.
_SAR_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["VV"] }],
        output: { bands: 1, sampleType: "AUTO" }
    };
}
function evaluatePixel(sample) {
    // Safety check
    if (!sample || sample.VV === undefined) {
        return [0];
    }
    // Convert linear → dB, scale [-30, 5] dB → [0, 1]
    var db = 10 * Math.log(sample.VV + 1e-6) / Math.log(10);
    var scaled = (db + 30) / 35.0;
    return [Math.max(0, Math.min(1, scaled))];
}
"""


def _adaptive_size(bbox_coords: list) -> tuple:
    """Return (width_px, height_px) targeting ~15 m/pixel, capped at 1024."""
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    clat     = (min_lat + max_lat) / 2
    w_km     = abs(max_lon - min_lon) * 111.0 * math.cos(math.radians(clat))
    h_km     = abs(max_lat - min_lat) * 111.0
    px_per_km = 66   # ~15 m/pixel
    w = int(min(1024, max(512, w_km * px_per_km)))
    h = int(min(1024, max(512, h_km * px_per_km)))
    return (w, h)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype in (np.float32, np.float64):
        return (arr * 255).clip(0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


# ── Optical Sentinel-2 ────────────────────────────────────────────────────────
def _optical_request(bbox: BBox, date_string: str, size: tuple, window_days: int = 365):
    """Request with a 1-year window to maximize chances of finding clear imagery."""
    end   = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    start = end - datetime.timedelta(days=window_days)
    return SentinelHubRequest(
        evalscript=_OPTICAL_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
        bbox=bbox, size=size, config=config,
    )


def fetch_satellite_image_bbox(bbox_coords: list, date_string: str) -> Image.Image:
    """
    Cloud-masked Sentinel-2 L2A image for an explicit bbox.
    Uses an intelligent cloud filter - prefers clear pixels but falls back to
    any available data if the entire region is cloudy.
    bbox_coords: [min_lon, min_lat, max_lon, max_lat]
    Returns best available image from a 1-year window.
    """
    print(f"📡 Optical [{date_string}] bbox={[round(x,4) for x in bbox_coords]}")
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    size = _adaptive_size(bbox_coords)
    print(f"   → image size: {size[0]}×{size[1]} px")
    
    try:
        arr = _optical_request(bbox, date_string, size).get_data()[0]
        
        # Check if image is all zeros (no data in this region/timeframe)
        if arr.max() == 0:
            print(f"⚠️  No data in 1-year window, trying 2-year window...")
            arr = _optical_request(bbox, date_string, size, window_days=730).get_data()[0]
            
            if arr.max() == 0:
                raise ValueError(
                    f"No Sentinel-2 coverage for this location/period. "
                    f"Try a different date or location. "
                    f"(Sentinel-2 operational since mid-2015)"
                )
        
        img = Image.fromarray(_to_uint8(arr))
        print("✅ Optical ready")
        return img
        
    except Exception as e:
        print(f"❌ Optical fetch failed: {e}")
        raise


# ── SAR Sentinel-1 (cloud-penetrating) ───────────────────────────────────────
def fetch_sar_image_bbox(bbox_coords: list, date_string: str) -> Image.Image | None:
    """
    Sentinel-1 IW GRD VV backscatter — cloud-penetrating, all-weather.
    Returns a grayscale PIL Image, or None if no data is available.
    bbox_coords: [min_lon, min_lat, max_lon, max_lat]
    """
    print(f"📡 SAR   [{date_string}] bbox={[round(x,4) for x in bbox_coords]}")
    try:
        bbox  = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        size  = _adaptive_size(bbox_coords)
        end   = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        start = end - datetime.timedelta(days=90)

        request = SentinelHubRequest(
            evalscript=_SAR_EVALSCRIPT,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW,
                    time_interval=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
                    mosaicking_order=MosaickingOrder.MOST_RECENT,
                )
            ],
            responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
            bbox=bbox, size=size, config=config,
        )
        arr = request.get_data()[0]
        # SAR returns single band — squeeze to 2D
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        img = Image.fromarray(_to_uint8(arr), mode='L')
        print("✅ SAR ready")
        return img
    except Exception as e:
        print(f"⚠️  SAR unavailable ({e}) — proceeding optical-only")
        return None


# ── Legacy point-based helper (kept for backward compat) ─────────────────────
def fetch_satellite_image(lat: float, lon: float, date_string: str, zoom: float = 0.15) -> Image.Image:
    coords = [lon - zoom, lat - zoom, lon + zoom, lat + zoom]
    return fetch_satellite_image_bbox(coords, date_string)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_bbox = [77.53, 12.93, 77.63, 13.01]   # Bengaluru central
    test_date = "2023-01-15"  # Known good date
    print("🚀 Smoke test …")
    try:
        opt = fetch_satellite_image_bbox(test_bbox, test_date)
        opt.save("test_optical.png")
        print("✅ Optical → test_optical.png")
        sar = fetch_sar_image_bbox(test_bbox, test_date)
        if sar:
            sar.save("test_sar.png")
            print("✅ SAR → test_sar.png")
    except Exception as e:
        print(f"❌ {e}")