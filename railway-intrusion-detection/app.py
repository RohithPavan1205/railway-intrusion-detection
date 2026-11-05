from pathlib import Path
import cv2
import pandas as pd
import streamlit as st

from config_runtime import BASE_DIR, deep_merge, normalize_config
from dashboard.runtime import ProcessingWorker
from dashboard.uploads import inspect_video, save_uploaded_video
from main import RailwayIntrusionSystem

st.set_page_config(
    page_title="RailGuard — Track Intrusion Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Overall page background */
    [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    [data-testid="stHeader"] { 
        background: transparent !important; 
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Force all text in sidebar to be clearly visible */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #0F172A;
    }

    .sb-heading {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B !important;
        margin: 0 0 0.6rem 0;
        display: block;
    }

    /* Streamlit Selectbox and Inputs Light Theme Override */
    [data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
        border-radius: 8px !important;
    }

    [data-baseweb="select"] * {
        color: #0F172A !important;
        background-color: transparent !important;
    }

    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
    }

    [data-baseweb="menu"] li {
        color: #0F172A !important;
    }

    /* Number inputs and steppers */
    [data-baseweb="input"] {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    [data-baseweb="input"] input {
        background-color: transparent !important;
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }

    div[data-testid="stNumberInput"] button svg {
        fill: #0F172A !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        height: 2.6rem !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2) !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
    }

    .stButton > button:disabled {
        background-color: #F1F5F9 !important;
        color: #94A3B8 !important;
        border-color: #E2E8F0 !important;
        opacity: 0.7 !important;
    }

    .stButton > button:disabled p,
    .stButton > button:disabled span {
        color: #94A3B8 !important;
    }

    /* Slider Styling */
    [data-testid="stSlider"] label {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    
    [data-baseweb="slider"] [role="slider"] {
        background-color: #2563EB !important;
    }

    /* Top Navigation / Status Header */
    .nav-bar {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.25rem;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }

    .nav-logo-badge {
        width: 38px;
        height: 38px;
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.95rem;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .nav-brand-text h1 {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .nav-brand-text p {
        font-size: 0.8rem;
        color: #64748B;
        margin: 2px 0 0 0;
    }

    .nav-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-standby {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }

    .status-running {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
    }

    .status-alert {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: currentColor;
    }

    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.35rem 0.8rem;
        text-align: right;
    }

    .stat-card .s-num {
        font-size: 1rem;
        font-weight: 700;
        color: #0F172A;
        font-variant-numeric: tabular-nums;
        line-height: 1.1;
    }

    .stat-card .s-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
    }

    /* Video Panel Cards */
    .feed-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
    }

    .feed-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0F172A;
    }

    .feed-tag {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
    }

    .tag-neutral {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
    }

    .tag-blue {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
    }

    .empty-placeholder {
        min-height: 380px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        text-align: center;
    }

    .empty-placeholder p {
        font-size: 0.92rem;
        font-weight: 600;
        color: #0F172A;
        margin: 0;
    }

    .empty-placeholder span {
        font-size: 0.8rem;
        color: #64748B;
        max-width: 260px;
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    /* Meta Info Box in Sidebar */
    .meta-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin: 0.5rem 0 0.85rem 0;
    }

    .meta-box .meta-name {
        font-size: 0.84rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 2px;
    }

    .meta-box .meta-details {
        font-size: 0.75rem;
        color: #64748B;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #F1F5F9;
        padding: 4px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748B;
        padding: 6px 14px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }

    .block-container {
        max-width: 1560px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session():
    defaults = {
        "system": None,
        "worker": None,
        "video": None,
        "upload_id": None,
        "current_preset": None,
        "pending_config": normalize_config({}),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def system_for_session():
    if st.session_state.system is None:
        st.session_state.system = RailwayIntrusionSystem(config=st.session_state.pending_config)
        st.session_state.worker = ProcessingWorker(st.session_state.system)
    return st.session_state.system


def processing_running():
    return bool(st.session_state.worker and st.session_state.worker.is_running)


def to_rgb(frame):
    return (
        cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if len(frame.shape) == 2
        else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )


initialize_session()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="sb-heading">Video Source</div>', unsafe_allow_html=True)

    sample_optical = BASE_DIR / "data" / "samples" / "optical_sample.mp4"
    if not sample_optical.exists():
        sample_optical = Path("data/samples/optical_sample.mp4")

    sample_thermal = BASE_DIR / "data" / "samples" / "thermal_sample.mp4"
    if not sample_thermal.exists():
        sample_thermal = Path("data/samples/thermal_sample.mp4")

    choices = {}
    if sample_optical.exists():
        choices["Daytime Demo (Optical)"] = str(sample_optical)
    if sample_thermal.exists():
        choices["Night-Vision Demo (Thermal)"] = str(sample_thermal)
    choices["Custom Upload"] = "custom"

    selected = st.selectbox(
        "Choose video feed",
        options=list(choices.keys()),
        disabled=processing_running(),
        label_visibility="collapsed",
    )

    preset_path = choices[selected]

    if preset_path != "custom":
        if st.session_state.current_preset != preset_path:
            st.session_state.video = inspect_video(preset_path)
            st.session_state.current_preset = preset_path
            st.session_state.upload_id = preset_path
    else:
        st.session_state.current_preset = "custom"
        uploaded = st.file_uploader(
            "Upload CCTV clip",
            type=["mp4", "avi", "mov"],
            disabled=processing_running(),
            label_visibility="collapsed",
        )
        if uploaded is not None:
            uid = (uploaded.name, uploaded.size)
            if uid != st.session_state.upload_id:
                meta = save_uploaded_video(uploaded)
                if meta["valid"]:
                    st.session_state.video = meta
                    st.session_state.upload_id = uid
                else:
                    st.error(meta["error"])

    # Video details box
    if st.session_state.video:
        v = st.session_state.video
        st.markdown(
            f"""
            <div class="meta-box">
                <div class="meta-name">{v['name']}</div>
                <div class="meta-details">{v['width']}x{v['height']} &bull; {v['duration_seconds']:.1f}s &bull; {v['size_bytes']/(1024*1024):.1f} MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="sb-heading">Pipeline Control</div>', unsafe_allow_html=True)

    has_video = bool(st.session_state.video)

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button(
            "Start",
            type="primary",
            disabled=processing_running() or not has_video,
            use_container_width=True,
        ):
            try:
                with st.spinner("Starting detector..."):
                    sys_ = system_for_session()
                    sys_.update_config(st.session_state.pending_config)
                    sys_.update_stream_source("optical", st.session_state.video["path"])
                    sys_.update_stream_source("thermal", "")
                    st.session_state.worker.start()
                st.rerun()
            except Exception as err:
                st.error(str(err))

    with col_stop:
        if st.button(
            "Stop",
            type="secondary",
            disabled=not processing_running(),
            use_container_width=True,
        ):
            if st.session_state.worker:
                st.session_state.worker.stop()
            st.rerun()

    if st.button(
        "Reset Tracker",
        disabled=processing_running(),
        use_container_width=True,
    ):
        if st.session_state.system:
            st.session_state.system.reset_session()
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sb-heading">Detection Settings</div>', unsafe_allow_html=True)

    cfg = st.session_state.pending_config
    conf_val = st.slider(
        "Confidence threshold",
        0.10, 0.95,
        float(cfg["models"]["optical"]["conf_threshold"]),
        0.05,
    )
    confirm_frames = st.number_input(
        "Confirmation frames",
        min_value=1, max_value=60,
        value=int(cfg["intrusion"]["temporal_confirmation_frames"]),
    )

    st.session_state.pending_config = normalize_config(
        deep_merge(cfg, {
            "models": {
                "optical": {"conf_threshold": conf_val},
                "thermal": {"conf_threshold": conf_val},
            },
            "intrusion": {"temporal_confirmation_frames": confirm_frames},
            "fusion": {"enabled": False},
        })
    )
    if st.session_state.system:
        st.session_state.system.update_config(st.session_state.pending_config)


# ==========================================
# MAIN PAGE TOP NAVBAR
# ==========================================

@st.fragment(run_every=1.0)
def render_navbar():
    system = st.session_state.system
    worker = st.session_state.worker
    running = bool(worker and worker.is_running)
    state = system.state if system else None

    fps = state["fps"] if running and state and state["processed_frames"] else 0.0
    lat_ms = state["latency"] * 1000 if running and state and state["latency"] else 0.0
    events = len(state["events"]) if state else 0
    tracks = len(state["tracks"]) if state else 0
    raw_dets = state.get("last_detections_count", 0) if state else 0
    frames_n = state.get("processed_frames", 0) if state else 0

    has_intrusion = (
        system
        and hasattr(system.intrusion_engine, "active_events")
        and len(system.intrusion_engine.active_events) > 0
    )

    if running and has_intrusion:
        status_html = '<div class="status-pill status-alert"><span class="status-dot"></span>Intrusion Detected</div>'
    elif running:
        status_html = '<div class="status-pill status-running"><span class="status-dot"></span>Monitoring</div>'
    else:
        status_html = '<div class="status-pill status-standby"><span class="status-dot"></span>Standby</div>'

    st.markdown(
        f"""
        <div class="nav-bar">
            <div class="nav-brand">
                <div class="nav-logo-badge">RG</div>
                <div class="nav-brand-text">
                    <h1>RailGuard</h1>
                    <p>Railway Track Intrusion Detection</p>
                </div>
            </div>
            <div class="nav-right">
                {status_html}
                <div class="stat-card">
                    <div class="s-num">{fps:.1f}</div>
                    <div class="s-label">FPS</div>
                </div>
                <div class="stat-card">
                    <div class="s-num">{lat_ms:.0f} ms</div>
                    <div class="s-label">Latency</div>
                </div>
                <div class="stat-card">
                    <div class="s-num" style="color: #2563EB;">{raw_dets}</div>
                    <div class="s-label">Detections</div>
                </div>
                <div class="stat-card">
                    <div class="s-num" style="color: #6366F1;">{tracks}</div>
                    <div class="s-label">Tracks</div>
                </div>
                <div class="stat-card">
                    <div class="s-num" style="color: {'#DC2626' if events > 0 else '#059669'};">{events}</div>
                    <div class="s-label">Alerts</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_navbar()


# ==========================================
# MAIN TABS
# ==========================================

tab_live, tab_events, tab_zones = st.tabs(["Live Feed", "Incident Log", "Safety Zones"])


# TAB 1: Live Feeds
with tab_live:
    col_raw, col_ai = st.columns(2, gap="medium")

    @st.fragment(run_every=1.0)
    def render_feeds():
        system = st.session_state.system
        state = system.state if system else None

        raw = None
        if state:
            raw = state["optical_frame"] if state["optical_frame"] is not None else state["thermal_frame"]
        annotated = state["fused_frame"] if state else None

        with col_raw:
            st.markdown(
                """
                <div class="feed-header">
                    <span class="feed-title">Source Video</span>
                    <span class="feed-tag tag-neutral">Input Feed</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if raw is not None:
                st.image(to_rgb(raw), use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="empty-placeholder">
                        <p>No video active</p>
                        <span>Select a video from the sidebar and click Start to begin monitoring.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_ai:
            st.markdown(
                """
                <div class="feed-header">
                    <span class="feed-title">Detection & Zone Overlay</span>
                    <span class="feed-tag tag-blue">YOLOv8 + Tracker</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if annotated is not None:
                st.image(to_rgb(annotated), use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="empty-placeholder">
                        <p>Awaiting detection</p>
                        <span>Real-time bounding boxes, tracked person IDs, and track boundaries will appear here.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_feeds()


# TAB 2: Incident Log
with tab_events:
    @st.fragment(run_every=1.0)
    def render_incidents():
        system = st.session_state.system
        state = system.state if system else None
        events = state["events"] if state else []

        hdr, dl_col = st.columns([5, 1])
        hdr.markdown(
            f"<h3 style='margin:0;font-size:1.1rem;color:#0F172A;'>Recorded Incidents ({len(events)})</h3>"
            f"<p style='margin:2px 0 12px 0;font-size:0.82rem;color:#64748B;'>Events logged when a tracked person enters the track zone polygon.</p>",
            unsafe_allow_html=True,
        )

        if not events:
            st.info("No intrusion events recorded in this session.")
            return

        df = pd.DataFrame(events)
        df.insert(0, "#", range(1, len(df) + 1))
        df["Time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        display = (
            df[["#", "Time", "person_id", "zone_name", "confidence"]]
            .rename(columns={"person_id": "Person ID", "zone_name": "Restricted Zone", "confidence": "Confidence"})
        )
        display["Confidence"] = display["Confidence"].map(lambda v: f"{v:.1%}")

        with dl_col:
            st.download_button(
                "Export CSV",
                df.to_csv(index=False).encode(),
                "incidents.csv",
                "text/csv",
                use_container_width=True,
            )

        st.dataframe(
            display.iloc[::-1],
            use_container_width=True,
            hide_index=True,
            height=240,
        )

        snap_dir = BASE_DIR / "outputs" / "snapshots"
        if not snap_dir.exists():
            snap_dir = Path("outputs/snapshots")
        if snap_dir.exists():
            snaps = sorted(snap_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
            if snaps:
                st.markdown(
                    "<p style='font-size:0.88rem;font-weight:600;color:#0F172A;margin-top:1.2rem;margin-bottom:0.5rem'>Intrusion Snapshots</p>",
                    unsafe_allow_html=True,
                )
                cols = st.columns(min(6, len(snaps)))
                for i, img_path in enumerate(snaps[:6]):
                    with cols[i]:
                        st.image(str(img_path), use_container_width=True)
                        st.caption(img_path.name[:20])

    render_incidents()


# TAB 3: Safety Zones
with tab_zones:
    st.markdown(
        "<h3 style='margin:0;font-size:1.1rem;color:#0F172A;'>Safety Zone Geometries</h3>"
        "<p style='margin:2px 0 16px 0;font-size:0.82rem;color:#64748B;'>Monitored polygon regions configured for intrusion triggers.</p>",
        unsafe_allow_html=True,
    )

    zones = st.session_state.pending_config["intrusion"]["restricted_zones"]
    for zone in zones:
        pts = " &rarr; ".join(f"({x},{y})" for x, y in zone["polygon"])
        with st.container(border=True):
            z1, z2 = st.columns([1, 2])
            with z1:
                st.markdown(f"**Zone Label:** `{zone['name']}`")
                st.markdown(f"**Vertices:** `{len(zone['polygon'])} points`")
                st.markdown("**Rule:** Foot-point polygon containment")
            with z2:
                st.markdown(
                    f"**Polygon Points:**<br/><div style='background:#F1F5F9;padding:8px 12px;border-radius:6px;font-family:monospace;font-size:0.82rem;color:#0F172A;margin-top:4px;'>{pts}</div>",
                    unsafe_allow_html=True,
                )
