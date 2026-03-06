import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import requests
import io
import math
import numpy as np
from PIL import Image

# ── Local modules ─────────────────────────────────────────────────────────────
from satellite_api import fetch_satellite_image_bbox, fetch_sar_image_bbox
from model_engine import generate_change_mask, overlay_mask


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bbox_from_drawing(drawing: dict) -> list | None:
    """Extract [min_lon, min_lat, max_lon, max_lat] from a Folium Draw GeoJSON feature."""
    try:
        coords = drawing["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [min(lons), min(lats), max(lons), max(lats)]
    except Exception:
        return None


def bbox_area_km2(bbox: list) -> float:
    """Rough area in km² of a [min_lon, min_lat, max_lon, max_lat] bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    h = abs(max_lat - min_lat) * 111.0
    w = abs(max_lon - min_lon) * 111.0 * math.cos(math.radians(center_lat))
    return round(h * w, 2)


@st.cache_data(show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> str:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "PixelBrains/1.0"},
            timeout=6,
        )
        data = r.json()
        addr = data.get("address", {})
        parts = [
            addr.get("suburb") or addr.get("city_district") or addr.get("village"),
            addr.get("city") or addr.get("town") or addr.get("county"),
            addr.get("state"),
            addr.get("country"),
        ]
        return ", ".join(p for p in parts if p) or data.get("display_name", f"{lat:.4f}°, {lon:.4f}°")
    except Exception:
        return f"{lat:.4f}°, {lon:.4f}°"


@st.cache_data(show_spinner=False)
def run_full_pipeline(bbox: list, start_date_str: str, end_date_str: str):
    """
    Fetch two cloud-free Sentinel-2 images (+ optional SAR) for a bbox,
    run multi-signal change detection, overlay red mask and return display data.
    Returns: (before_img, after_img, overlay_img, before_sar, after_sar,
              pct_changed, changed_km2, total_km2)
    """
    before_img = fetch_satellite_image_bbox(bbox, start_date_str)
    after_img  = fetch_satellite_image_bbox(bbox, end_date_str)

    # SAR (Sentinel-1 radar) — works through clouds; None if unavailable
    before_sar = fetch_sar_image_bbox(bbox, start_date_str)
    after_sar  = fetch_sar_image_bbox(bbox, end_date_str)

    before_sar_bytes = pil_to_bytes(before_sar) if before_sar else None
    after_sar_bytes  = pil_to_bytes(after_sar)  if after_sar  else None

    mask_img    = generate_change_mask(
        pil_to_bytes(before_img), pil_to_bytes(after_img),
        before_sar_bytes, after_sar_bytes,
    )
    overlay_img = overlay_mask(after_img, mask_img, color=(255, 20, 20), alpha=0.65)

    mask_arr    = np.array(mask_img.convert("L"))
    pct_changed = round(float((mask_arr > 127).sum()) / mask_arr.size * 100, 2)

    total_km2   = bbox_area_km2(bbox)
    changed_km2 = round(total_km2 * pct_changed / 100, 2)

    return before_img, after_img, overlay_img, before_sar, after_sar, pct_changed, changed_km2, total_km2

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pixel Brains – Satellite Uplink",
    page_icon="🛰️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* dark, space-like background */
        .stApp { background-color: #0d1117; color: #e6edf3; }

        /* glowing title */
        h1 { color: #58a6ff; text-align: center; letter-spacing: 3px; }
        h3 { color: #8b949e; text-align: center; }

        /* coordinate display box */
        .coord-box {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 20px;
            margin-top: 8px;
            font-family: monospace;
            font-size: 15px;
            color: #58a6ff;
        }

        /* result card */
        .result-card {
            background: #161b22;
            border: 1px solid #238636;
            border-radius: 10px;
            padding: 20px 28px;
            margin-top: 16px;
        }
        .result-card h4 { color: #3fb950; margin-bottom: 12px; }
        .result-card p  { color: #c9d1d9; font-size: 15px; margin: 6px 0; }
        .result-card span { color: #79c0ff; font-weight: bold; }

        /* map section label */
        .map-label {
            color: #8b949e;
            font-size: 13px;
            margin-bottom: 4px;
        }

        /* Streamlit button override */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #238636, #2ea043);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 16px;
            letter-spacing: 2px;
            font-weight: 700;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        div.stButton > button:hover { opacity: 0.85; }

        /* text inputs */
        div[data-testid="stTextInput"] input {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #e6edf3;
        }

        /* hide Streamlit header/footer */
        #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>🛰️ PIXEL BRAINS — SATELLITE UPLINK</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3>Select a target zone on the map · Set observation window · Initiate scan</h3>",
    unsafe_allow_html=True,
)
st.divider()

# ── Session state defaults ────────────────────────────────────────────────────
for key in ("bbox", "center", "live_bbox", "live_center"):
    if key not in st.session_state:
        st.session_state[key] = None

BENGALURU = [12.9716, 77.5946]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🗺️ Change Detection", "📊 Analysis Results", "🛰️ Live Global Uplink"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Change Detection (original content)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    map_col, ctrl_col = st.columns([3, 1], gap="large")

    with map_col:
        st.markdown(
            '<p class="map-label">🟦 Draw a rectangle on the map to select your analysis zone</p>',
            unsafe_allow_html=True,
        )

        m1 = folium.Map(
            location=BENGALURU,
            zoom_start=10,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            control_scale=True,
        )

        # Draw plugin — rectangle only
        Draw(
            draw_options={
                "rectangle": {"shapeOptions": {"color": "#58a6ff", "weight": 2}},
                "polygon": False, "circle": False, "marker": False,
                "polyline": False, "circlemarker": False,
            },
            edit_options={"edit": False, "remove": True},
        ).add_to(m1)

        # Render previously drawn bbox
        if st.session_state.bbox:
            mn, mi_lat, mx, ma_lat = st.session_state.bbox
            folium.Rectangle(
                bounds=[[mi_lat, mn], [ma_lat, mx]],
                color="#58a6ff", fill=True, fill_opacity=0.1, weight=2,
            ).add_to(m1)

        map1_data = st_folium(
            m1, key="tab1_map",
            use_container_width=True, height=520,
            returned_objects=["all_drawings"],
        )

        # Capture drawn rectangle ONLY if not currently processing
        # (prevents bbox from being cleared on button click)
        if "processing_t1" not in st.session_state:
            st.session_state.processing_t1 = False
        
        if not st.session_state.processing_t1:
            drawings = (map1_data or {}).get("all_drawings") or []
            if drawings:
                bbox = bbox_from_drawing(drawings[-1])
                if bbox and bbox != st.session_state.bbox:
                    mn_lon, mn_lat, mx_lon, mx_lat = bbox
                    center = [(mn_lat + mx_lat) / 2, (mn_lon + mx_lon) / 2]
                    st.session_state.bbox   = bbox
                    st.session_state.center = center
                    st.rerun()

    with ctrl_col:
        st.markdown("### 🎯 Selected Zone")

        if st.session_state.bbox:
            mn_lon, mn_lat, mx_lon, mx_lat = st.session_state.bbox
            clat, clon = st.session_state.center
            total_km2 = bbox_area_km2(st.session_state.bbox)
            st.markdown(
                f'<div class="coord-box">'
                f'<b>SW</b> {mn_lat:.4f}°, {mn_lon:.4f}°<br>'
                f'<b>NE</b> {mx_lat:.4f}°, {mx_lon:.4f}°<br>'
                f'<b>Area</b> ~{total_km2} km²'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="coord-box" style="color:#484f58;">'
                '⬅ Draw a rectangle on the map'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📅 Year Window")
        st.caption("Sentinel-2 available from 2015 onwards")

        start_year = st.text_input("Start Year", placeholder="e.g. 2016", max_chars=4, key="t1_start_year")
        end_year   = st.text_input("End Year",   placeholder="e.g. 2024", max_chars=4, key="t1_end_year")

        st.markdown("---")
        uplink_btn = st.button("⚡ INITIATE SATELLITE UPLINK", key="t1_uplink_btn")
        if st.button("🗑️ Clear Selection", key="t1_clear_btn"):
            st.session_state.bbox   = None
            st.session_state.center = None
            st.rerun()

    # ── Result panel ─────────────────────────────────────────────────────────
    if uplink_btn:
        st.session_state.processing_t1 = True
        errors = []
        if not st.session_state.bbox:
            errors.append("No zone selected — draw a rectangle on the map first.")
        if not start_year.strip():
            errors.append("Start Year is required.")
        if not end_year.strip():
            errors.append("End Year is required.")
        if start_year.strip() and end_year.strip():
            try:
                sy, ey = int(start_year.strip()), int(end_year.strip())
                if sy < 2015:
                    errors.append("Sentinel-2 data starts from 2015.")
                if sy >= ey:
                    errors.append("Start Year must be earlier than End Year.")
            except ValueError:
                errors.append("Years must be numeric (e.g. 2016).")

        if errors:
            for err in errors:
                st.error(err)
        else:
            sy, ey   = int(start_year.strip()), int(end_year.strip())
            bbox     = st.session_state.bbox
            clat, clon = st.session_state.center

            # Use July 1 as mid-year clear-sky anchor
            start_date_str = f"{sy}-07-01"
            end_date_str   = f"{ey}-07-01"

            with st.spinner("🛰️ Fetching satellite imagery & running change detection…"):
                place_name = reverse_geocode(clat, clon)
                try:
                    before_img, after_img, overlay_img, before_sar, after_sar, pct_changed, changed_km2, total_km2 = \
                        run_full_pipeline(bbox, start_date_str, end_date_str)
                    pipeline_ok = True
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    pipeline_ok = False
                finally:
                    st.session_state.processing_t1 = False
            if pipeline_ok:
                mn_lon, mn_lat, mx_lon, mx_lat = bbox
                st.markdown(
                    f'<div class="result-card" style="border-color:#58a6ff; margin-bottom:16px;">'
                    f'<h4 style="color:#58a6ff;">📍 Target Location</h4>'
                    f'<p style="font-size:17px;">{place_name}</p>'
                    f'<p>🌐 <b>Zone</b> [{mn_lat:.4f}°, {mn_lon:.4f}°] → [{mx_lat:.4f}°, {mx_lon:.4f}°]</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3, gap="small")
                with c1:
                    st.markdown(f"**📷 Before — {sy}**")
                    st.image(before_img, use_container_width=True)
                with c2:
                    st.markdown(f"**📷 After — {ey}**")
                    st.image(after_img, use_container_width=True)
                with c3:
                    st.markdown("**🔴 Change Detected**")
                    st.image(overlay_img, use_container_width=True)

                # Show SAR imagery if available
                if before_sar is not None or after_sar is not None:
                    st.markdown("---")
                    st.markdown("**📡 SAR Radar Imagery (Sentinel-1 — cloud-penetrating)**")
                    s1, s2 = st.columns(2, gap="small")
                    with s1:
                        if before_sar:
                            st.markdown(f"SAR Before — {sy}")
                            st.image(before_sar, use_container_width=True)
                        else:
                            st.caption("SAR unavailable for start date")
                    with s2:
                        if after_sar:
                            st.markdown(f"SAR After — {ey}")
                            st.image(after_sar, use_container_width=True)
                        else:
                            st.caption("SAR unavailable for end date")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4>📊 Change Detection Report</h4>
                        <p>📍 <b>Location &nbsp;&nbsp;&nbsp;&nbsp;:</b> <span>{place_name}</span></p>
                        <p>📅 <b>Period &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</b> <span>{sy} → {ey} &nbsp;({ey-sy} yr)</span></p>
                        <p>🔴 <b>Changed Area&nbsp;:</b> <span>{pct_changed}% &nbsp;(~{changed_km2} km²)</span></p>
                        <p>📐 <b>Total Zone &nbsp;&nbsp;:</b> <span>~{total_km2} km²</span></p>
                        <p>🌐 <b>Centre &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</b> <span>{clat:.4f}° N, {clon:.4f}° E</span></p>
                        <p>📡 <b>SAR Signal &nbsp;&nbsp;:</b> <span>{'Active ✅' if before_sar or after_sar else 'Unavailable ⚠️'}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analysis Results (placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="result-card" style="border-color:#484f58; text-align:center; padding:60px;">'
        '<h4 style="color:#8b949e;">📊 Analysis Results</h4>'
        '<p style="color:#484f58;">Run the change-detection pipeline from the <b>Change Detection</b> tab<br>'
        'to populate results here.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Global Uplink
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🛰️ Live Global Uplink")
    st.markdown(
        '<p class="map-label">� Draw a rectangle anywhere on Earth to lock on to a live target zone</p>',
        unsafe_allow_html=True,
    )

    live_map = folium.Map(
        location=BENGALURU,
        zoom_start=4,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        control_scale=True,
    )

    Draw(
        draw_options={
            "rectangle": {"shapeOptions": {"color": "#3fb950", "weight": 2}},
            "polygon": False, "circle": False, "marker": False,
            "polyline": False, "circlemarker": False,
        },
        edit_options={"edit": False, "remove": True},
    ).add_to(live_map)

    if st.session_state.live_bbox:
        mn, mi_lat, mx, ma_lat = st.session_state.live_bbox
        folium.Rectangle(
            bounds=[[mi_lat, mn], [ma_lat, mx]],
            color="#3fb950", fill=True, fill_opacity=0.1, weight=2,
        ).add_to(live_map)

    live_map_data = st_folium(
        live_map, key="tab3_map",
        use_container_width=True, height=480,
        returned_objects=["all_drawings"],
    )

    # Capture drawn rectangle ONLY if not currently processing
    if "processing_t3" not in st.session_state:
        st.session_state.processing_t3 = False
    
    if not st.session_state.processing_t3:
        live_drawings = (live_map_data or {}).get("all_drawings") or []
        if live_drawings:
            lbbox = bbox_from_drawing(live_drawings[-1])
            if lbbox and lbbox != st.session_state.live_bbox:
                mn_lon, mn_lat, mx_lon, mx_lat = lbbox
                st.session_state.live_bbox   = lbbox
                st.session_state.live_center = [(mn_lat + mx_lat) / 2, (mn_lon + mx_lon) / 2]
                st.rerun()

    # ── Zone readout ───────────────────────────────────────────────────────────
    if st.session_state.live_bbox:
        mn_lon, mn_lat, mx_lon, mx_lat = st.session_state.live_bbox
        total_km2 = bbox_area_km2(st.session_state.live_bbox)
        st.markdown(
            f'<div class="coord-box">'
            f'🌐 <b>SW</b> {mn_lat:.4f}°, {mn_lon:.4f}°'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;<b>NE</b> {mx_lat:.4f}°, {mx_lon:.4f}°'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;<b>Zone</b> ~{total_km2} km²'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="coord-box" style="color:#484f58;">⬆ Draw a rectangle to select a live target zone</div>',
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear Selection", key="t3_clear_btn"):
        st.session_state.live_bbox   = None
        st.session_state.live_center = None
        st.rerun()

    st.markdown("---")

    date_col1, date_col2, btn_col = st.columns([2, 2, 1], gap="medium")
    with date_col1:
        live_start_date = st.text_input(
            "Start Date (YYYY-MM-DD)", placeholder="e.g. 2018-01-01", key="live_start_date"
        )
    with date_col2:
        live_end_date = st.text_input(
            "End Date (YYYY-MM-DD)", placeholder="e.g. 2024-12-01", key="live_end_date"
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        live_scan_btn = st.button("🔭 Initiate Live Scan", key="live_scan_btn")

    # ── Scan result ───────────────────────────────────────────────────────────
    if live_scan_btn:
        st.session_state.processing_t3 = True
        live_errors = []
        if not st.session_state.live_bbox:
            live_errors.append("No target zone selected — draw a rectangle on the map first.")
        if not live_start_date.strip():
            live_errors.append("Start Date is required.")
        if not live_end_date.strip():
            live_errors.append("End Date is required.")

        if not live_errors:
            try:
                from datetime import datetime as _dt
                sd = _dt.strptime(live_start_date.strip(), "%Y-%m-%d")
                ed = _dt.strptime(live_end_date.strip(),   "%Y-%m-%d")
                if sd.year < 2015:
                    live_errors.append("Sentinel-2 data starts from 2015.")
                if sd >= ed:
                    live_errors.append("Start Date must be before End Date.")
            except ValueError:
                live_errors.append("Dates must be in YYYY-MM-DD format.")

        if live_errors:
            for err in live_errors:
                st.error(err)
        else:
            lbbox   = st.session_state.live_bbox
            lc      = st.session_state.live_center
            llat, llon = lc

            with st.spinner("🛰️ Fetching live satellite imagery & running change detection…"):
                place_name = reverse_geocode(llat, llon)
                try:
                    before_img, after_img, overlay_img, before_sar, after_sar, pct_changed, changed_km2, total_km2 = \
                        run_full_pipeline(lbbox, live_start_date.strip(), live_end_date.strip())
                    live_ok = True
                except Exception as e:
                    st.error(f"Live scan failed: {e}")
                    live_ok = False
                finally:
                    st.session_state.processing_t3 = False
            if live_ok:
                mn_lon, mn_lat, mx_lon, mx_lat = lbbox
                st.markdown(
                    f'<div class="result-card" style="border-color:#58a6ff; margin-bottom:16px;">'
                    f'<h4 style="color:#58a6ff;">📍 Target Location</h4>'
                    f'<p style="font-size:17px;">{place_name}</p>'
                    f'<p>🌐 <b>Zone</b> [{mn_lat:.4f}°, {mn_lon:.4f}°] → [{mx_lat:.4f}°, {mx_lon:.4f}°]</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3, gap="small")
                with c1:
                    st.markdown(f"**📷 Before — {live_start_date.strip()}**")
                    st.image(before_img, use_container_width=True)
                with c2:
                    st.markdown(f"**📷 After — {live_end_date.strip()}**")
                    st.image(after_img, use_container_width=True)
                with c3:
                    st.markdown("**🔴 Change Detected**")
                    st.image(overlay_img, use_container_width=True)

                # SAR imagery
                if before_sar is not None or after_sar is not None:
                    st.markdown("---")
                    st.markdown("**📡 SAR Radar Imagery (Sentinel-1 — cloud-penetrating)**")
                    s1, s2 = st.columns(2, gap="small")
                    with s1:
                        if before_sar:
                            st.markdown(f"SAR Before — {live_start_date.strip()}")
                            st.image(before_sar, use_container_width=True)
                        else:
                            st.caption("SAR unavailable for start date")
                    with s2:
                        if after_sar:
                            st.markdown(f"SAR After — {live_end_date.strip()}")
                            st.image(after_sar, use_container_width=True)
                        else:
                            st.caption("SAR unavailable for end date")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4>📊 Live Scan Report</h4>
                        <p>📍 <b>Location &nbsp;&nbsp;&nbsp;&nbsp;:</b> <span>{place_name}</span></p>
                        <p>📅 <b>Period &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</b> <span>{live_start_date.strip()} → {live_end_date.strip()}</span></p>
                        <p>🔴 <b>Changed Area&nbsp;:</b> <span>{pct_changed}% &nbsp;(~{changed_km2} km²)</span></p>
                        <p>📐 <b>Total Zone &nbsp;&nbsp;:</b> <span>~{total_km2} km²</span></p>
                        <p>🌐 <b>Centre &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</b> <span>{llat:.4f}° N, {llon:.4f}° E</span></p>
                        <p>📡 <b>SAR Signal &nbsp;&nbsp;:</b> <span>{'Active ✅' if before_sar or after_sar else 'Unavailable ⚠️'}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )