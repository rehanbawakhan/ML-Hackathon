"""PathSnap evaluator UI (Laptop 2).

Talks only to the API over the network: lists the evaluation trips, sends the selected
trip ID, and shows the returned predicted route and metrics on a Folium map.

Run:
    streamlit run ui/app.py
Then enter the API address of Laptop 1, e.g. http://192.168.1.20:8000
"""
from __future__ import annotations

import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap Evaluator", layout="wide")
st.title("PathSnap — Map-Matching Evaluator")

api = st.sidebar.text_input("API address (Laptop 1)", value=st.session_state.get("api", "http://127.0.0.1:8000"))
st.session_state["api"] = api.rstrip("/")

if st.sidebar.button("Connect"):
    try:
        h = requests.get(f"{st.session_state['api']}/health", timeout=5).json()
        st.session_state["trips"] = requests.get(f"{st.session_state['api']}/trips", timeout=5).json()["trips"]
        st.sidebar.success(f"Connected · model {h['model']} · {h['n_trips']} trips")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Cannot reach API: {e}")

trips = st.session_state.get("trips", [])
if not trips:
    st.info("Enter the API address and click Connect.")
    st.stop()

trip_id = st.selectbox("Evaluation trip", trips)
if st.button("Run map-matching", type="primary"):
    t0 = time.time()
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        resp = requests.post(f"{st.session_state['api']}/predict", json={"trip_id": trip_id}, timeout=600)
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        st.stop()
    st.session_state["result"] = resp.json()
    st.session_state["roundtrip"] = time.time() - t0

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    st.subheader(f"Trip {res['trip_id']} · {res['n_points']} GPS points")
    c = st.columns(5)
    if "point_accuracy_hmm" in m:
        c[0].metric("Point accuracy (HMM)", f"{m['point_accuracy_hmm']:.1%}",
                    f"{m['point_accuracy_hmm'] - m['point_accuracy_classifier_only']:+.1%} vs classifier")
        c[1].metric("Route F1", f"{m['route_f1']:.3f}")
    c[2].metric("Route continuity", f"{m['route_continuity_hmm']:.1%}")
    c[3].metric("API inference", f"{m['inference_seconds']:.1f} s")
    c[4].metric("Round trip", f"{st.session_state['roundtrip']:.1f} s")

    gps = res["gps"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles=None)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Basemap (Esri)").add_to(fmap)
    net = folium.FeatureGroup(name="Road network (grey)")
    for seg in res["network"]:
        folium.PolyLine(seg, color="#9e9e9e", weight=1, opacity=0.6).add_to(net)
    net.add_to(fmap)
    pred = folium.FeatureGroup(name="Predicted route (red)")
    for seg in res["predicted_route"]:
        folium.PolyLine(seg, color="#d32f2f", weight=5, opacity=0.85).add_to(pred)
    pred.add_to(fmap)
    pts = folium.FeatureGroup(name="GPS observations (blue)")
    for lat, lon in gps:
        folium.CircleMarker((lat, lon), radius=2, color="#1565c0", fill=True, fill_opacity=0.8).add_to(pts)
    pts.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[])

    with st.expander("Full metrics from API"):
        st.json(m)
