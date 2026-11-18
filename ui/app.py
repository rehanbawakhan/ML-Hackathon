"""PathSnap evaluator UI (Laptop 2).

Talks only to the API over the network: lists the evaluation trips, sends the selected
trip ID, and shows the returned predicted route and metrics on a Folium map.

Run:
    python -m streamlit run ui/app.py
Then enter the API address of Laptop 1, e.g. http://192.168.1.20:8000
"""
from __future__ import annotations

import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🛰️", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, .stSelectbox, .stTextInput { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b2a4a 0%, #0d1117 55%) fixed; }
[data-testid="stSidebar"] { background: #0b0f17; border-right: 1px solid #1f2937; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1400px; }

.hero { display:flex; align-items:center; justify-content:space-between; gap:24px;
        padding: 26px 30px; border-radius: 18px; margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(56,189,248,.12), rgba(239,68,68,.10));
        border: 1px solid rgba(148,163,184,.18); }
.hero h1 { margin:0; font-size: 2.3rem; font-weight: 800; letter-spacing:-.02em; color:#f8fafc; }
.hero h1 span { background: linear-gradient(90deg,#38bdf8,#f87171); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { margin:6px 0 0; color:#94a3b8; font-size: 1rem; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
         background: rgba(148,163,184,.12); color:#cbd5e1; border:1px solid rgba(148,163,184,.25); margin: 0 6px 6px 0; }

.pipe { display:flex; align-items:stretch; gap:10px; margin: 6px 0 22px; flex-wrap: wrap; }
.step { flex:1; min-width:170px; padding:14px 16px; border-radius:14px; background:#111827; border:1px solid #1f2937; }
.step .n { font-family:'JetBrains Mono',monospace; color:#38bdf8; font-size:.8rem; }
.step .t { color:#f1f5f9; font-weight:700; margin-top:2px; }
.step .d { color:#94a3b8; font-size:.85rem; margin-top:4px; line-height:1.35; }
.arrow { display:flex; align-items:center; color:#475569; font-size:1.3rem; }

.card { padding:18px 20px; border-radius:16px; background:#111827; border:1px solid #1f2937; height:100%; }
.card .k { color:#94a3b8; font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.card .v { color:#f8fafc; font-size:2rem; font-weight:800; margin-top:4px; letter-spacing:-.02em; }
.card .s { font-size:.85rem; margin-top:2px; color:#94a3b8; }
.card .up { color:#4ade80; font-weight:700; }
.card.accent { border-color: rgba(74,222,128,.45); box-shadow: 0 0 0 1px rgba(74,222,128,.15) inset; }

.cmp { padding:18px 20px; border-radius:16px; background:#111827; border:1px solid #1f2937; }
.cmp h4 { margin:0 0 12px; color:#f1f5f9; font-size:1rem; }
.row { margin: 10px 0; }
.row .lbl { display:flex; justify-content:space-between; color:#cbd5e1; font-size:.85rem; margin-bottom:5px; }
.bar { height:10px; border-radius:999px; background:#1f2937; overflow:hidden; }
.bar > div { height:100%; border-radius:999px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; color:#cbd5e1; font-size:.85rem; margin: 4px 0 10px; }
.dot { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }

.stButton button[kind="primary"] { background: linear-gradient(90deg,#ef4444,#f97316); border:0; font-weight:700;
        padding: .6rem 1.4rem; border-radius: 12px; box-shadow: 0 8px 24px rgba(239,68,68,.25); }
.stButton button[kind="primary"]:hover { filter: brightness(1.08); }
.stButton button[kind="secondary"] { border-radius: 10px; width: 100%; font-weight:600; }
.side-h { color:#f1f5f9; font-weight:800; font-size:1.2rem; margin-bottom: 2px; }
.side-s { color:#64748b; font-size:.82rem; margin-bottom: 18px; }
iframe { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", accent=False):
    return f'<div class="card{" accent" if accent else ""}"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def bar_row(label, a, b):
    return f'''<div class="row"><div class="lbl"><span>{label}</span><span>{a:.1%} → <b style="color:#4ade80">{b:.1%}</b></span></div>
    <div class="bar" style="margin-bottom:4px"><div style="width:{a*100:.1f}%;background:#64748b"></div></div>
    <div class="bar"><div style="width:{b*100:.1f}%;background:linear-gradient(90deg,#22c55e,#4ade80)"></div></div></div>'''


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="side-h">🛰️ PathSnap</div><div class="side-s">Evaluator console · Laptop 2</div>', unsafe_allow_html=True)
    api = st.text_input("API address (Laptop 1)", value=st.session_state.get("api", "http://127.0.0.1:8000"))
    st.session_state["api"] = api.rstrip("/")
    if st.button("Connect", type="secondary"):
        try:
            h = requests.get(f"{st.session_state['api']}/health", timeout=5).json()
            st.session_state["trips"] = requests.get(f"{st.session_state['api']}/trips", timeout=5).json()["trips"]
            st.session_state["health"] = h
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("health", None)
            st.error(f"Cannot reach API: {e}")
    h = st.session_state.get("health")
    if h:
        st.success(f"● Connected\n\nModel: `{h['model']}`\n\nEvaluation trips: **{h['n_trips']}**")
    st.markdown("---")
    st.caption("Ground truth stays on Laptop 1. It is used only for scoring, after prediction, and is never sent here.")

# ----------------------------------------------------------------- header
st.markdown("""
<div class="hero"><div>
  <h1><span>PathSnap</span> Map-Matching Evaluator</h1>
  <p>Noisy GPS → road arcs → connected route. LightGBM candidate scoring + HMM/Viterbi decoding.</p>
  <div style="margin-top:12px">
    <span class="badge">UE24CS352A · ML Hackathon</span><span class="badge">PES University</span>
    <span class="badge">FastAPI ⇄ Streamlit</span>
  </div>
</div></div>
<div class="pipe">
  <div class="step"><div class="n">01</div><div class="t">Candidates</div><div class="d">Nearest 8 arcs within 75 m of each GPS point</div></div>
  <div class="arrow">›</div>
  <div class="step"><div class="n">02</div><div class="t">Classifier</div><div class="d">Tuned LightGBM scores each candidate (21 features)</div></div>
  <div class="arrow">›</div>
  <div class="step"><div class="n">03</div><div class="t">HMM / Viterbi</div><div class="d">Road-graph transitions enforce a connected route</div></div>
  <div class="arrow">›</div>
  <div class="step"><div class="n">04</div><div class="t">Score</div><div class="d">API compares with ground truth after prediction</div></div>
</div>
""", unsafe_allow_html=True)

trips = st.session_state.get("trips", [])
if not trips:
    st.info("Enter the API address of Laptop 1 in the sidebar and click **Connect**.")
    st.stop()

c1, c2 = st.columns([4, 1])
trip_id = c1.selectbox("Evaluation trip", trips, label_visibility="collapsed")
run = c2.button("▶  Run map-matching", type="primary", use_container_width=True)
if run:
    t0 = time.time()
    with st.spinner(f"Matching trip {trip_id} on Laptop 1..."):
        resp = requests.post(f"{st.session_state['api']}/predict", json={"trip_id": trip_id}, timeout=600)
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        st.stop()
    st.session_state["result"] = resp.json()
    st.session_state["roundtrip"] = time.time() - t0

res = st.session_state.get("result")
if not res:
    st.stop()

m = res["metrics"]
st.markdown(f"### Trip `{res['trip_id']}` &nbsp;·&nbsp; {res['n_points']:,} GPS points")

cols = st.columns(5)
if "point_accuracy_hmm" in m:
    gain = m["point_accuracy_hmm"] - m["point_accuracy_classifier_only"]
    cols[0].markdown(card("Point accuracy", f"{m['point_accuracy_hmm']:.1%}", f'<span class="up">{gain:+.1%}</span> vs classifier only', True), unsafe_allow_html=True)
    cols[1].markdown(card("Route F1", f"{m['route_f1']:.3f}", f"P {m['route_precision']:.3f} · R {m['route_recall']:.3f}"), unsafe_allow_html=True)
else:
    cols[0].markdown(card("Point accuracy", "n/a", "no ground truth"), unsafe_allow_html=True)
    cols[1].markdown(card("Route F1", "n/a", "no ground truth"), unsafe_allow_html=True)
cols[2].markdown(card("Continuity", f"{m['route_continuity_hmm']:.1%}", f"{m.get('hmm_breaks', 0)} chain restarts"), unsafe_allow_html=True)
cols[3].markdown(card("Inference", f"{m['inference_seconds']:.1f}s", "on Laptop 1"), unsafe_allow_html=True)
cols[4].markdown(card("Round trip", f"{st.session_state['roundtrip']:.1f}s", "request → response"), unsafe_allow_html=True)

st.write("")
left, right = st.columns([3, 1.15])

with right:
    html = '<div class="cmp"><h4>Classifier only vs + HMM</h4>'
    if "point_accuracy_hmm" in m:
        html += bar_row("Point accuracy", m["point_accuracy_classifier_only"], m["point_accuracy_hmm"])
    if "route_continuity_classifier_only" in m:
        html += bar_row("Route continuity", m["route_continuity_classifier_only"], m["route_continuity_hmm"])
    html += ('<div class="legend" style="margin-top:14px"><span><span class="dot" style="background:#64748b"></span>Classifier</span>'
             '<span><span class="dot" style="background:#4ade80"></span>+ HMM</span></div></div>')
    st.markdown(html, unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="cmp"><h4>Map legend</h4><div class="legend" style="flex-direction:column;gap:8px">'
                '<span><span class="dot" style="background:#f43f5e"></span>Predicted route</span>'
                '<span><span class="dot" style="background:#38bdf8"></span>GPS observations</span>'
                '<span><span class="dot" style="background:#475569"></span>Road network</span>'
                '<span><span class="dot" style="background:#22c55e"></span>Start &nbsp; <span class="dot" style="background:#ef4444;margin-left:6px"></span>End</span>'
                '</div></div>', unsafe_allow_html=True)

with left:
    gps = res["gps"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(fmap)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Streets (Esri)").add_to(fmap)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Satellite (Esri)").add_to(fmap)
    net = folium.FeatureGroup(name="Road network")
    for seg in res["network"]:
        folium.PolyLine(seg, color="#64748b", weight=1, opacity=0.55).add_to(net)
    net.add_to(fmap)
    pred = folium.FeatureGroup(name="Predicted route")
    for seg in res["predicted_route"]:
        folium.PolyLine(seg, color="#f43f5e", weight=6, opacity=0.9).add_to(pred)
    pred.add_to(fmap)
    pts = folium.FeatureGroup(name="GPS observations")
    for lat, lon in gps:
        folium.CircleMarker((lat, lon), radius=2, color="#38bdf8", weight=0, fill=True, fill_opacity=0.9).add_to(pts)
    pts.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    st_folium(fmap, height=600, use_container_width=True, returned_objects=[])

with st.expander("Raw metrics from API"):
    st.json(m)
