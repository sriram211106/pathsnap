"""Step 12 - PathSnap evaluator UI (runs on LAPTOP 2 only; no model, no data, no ground truth here).

Start:
    streamlit run ui/app.py
Then enter Laptop 1's API address in the sidebar, e.g. http://192.168.1.23:8000

Flow: list trips from the API -> evaluator picks one -> UI sends the trip ID to POST /predict
-> shows the predicted route on a map + the metrics computed by the API.
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching & route reconstruction")

api = st.sidebar.text_input("API address (Laptop 1)", value=os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000"))
api = api.rstrip("/")


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(base):
    r = requests.get(f"{base}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


try:
    health = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"Connected: model {health['model']}")
    trips = get_trips(api)
except Exception as e:
    st.sidebar.error(f"Cannot reach the API at {api}\n\n{e}")
    st.stop()

trip = st.selectbox(f"Evaluation trip ({len(trips)} available)", trips)
if st.button("Reconstruct route", type="primary"):
    with st.spinner(f"Sending trip {trip} to the API..."):
        try:
            resp = requests.post(f"{api}/predict", json={"trip_id": trip}, timeout=600)
            resp.raise_for_status()
            st.session_state["result"] = resp.json()
        except Exception as e:
            st.error(f"Request failed: {e}")

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    st.subheader(f"Trip {res['trip_id']}  ·  model {res['model']}")
    if m.get("available"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy (HMM)", f"{m['accuracy_hmm']:.2%}",
                  delta=f"{(m['accuracy_hmm'] - m['accuracy_classifier_only']) * 100:+.1f} pts vs classifier only")
        c2.metric("Accuracy (classifier only)", f"{m['accuracy_classifier_only']:.2%}")
        c3.metric("Route continuity (HMM)", f"{m['route_continuity_hmm']:.3f}")
        c4.metric("Points scored / total", f"{m['n_scored_points']:,} / {m['n_points']:,}")
        st.caption(m["note"])
    else:
        st.info("No reference available for this trip: prediction only.")
    st.caption("Timing (s): " + ", ".join(f"{k} {v}" for k, v in res["timing_s"].items()))

    gps = res["gps_points"]
    fmap = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri", name="Esri street map").add_to(fmap)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri", name="Esri satellite").add_to(fmap)
    step = max(1, len(gps) // 4000)
    g = folium.FeatureGroup(name=f"GPS observations ({len(gps):,}{', sampled' if step > 1 else ''})")
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2.5, color="#1f4fd1", fill=True, fill_opacity=0.8, weight=0).add_to(g)
    g.add_to(fmap)
    folium.PolyLine(res["predicted_route"], color="#d62728", weight=5, opacity=0.85,
                    name="Predicted route (HMM)").add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(fmap)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;'
        'border:1px solid #999;border-radius:6px;font-size:13px;">'
        '<span style="color:#1f4fd1;">&#9679;</span> GPS observation<br>'
        '<span style="color:#d62728;">&#9644;</span> Predicted route</div>'))
    try:
        st_folium(fmap, height=620, use_container_width=True, returned_objects=[])
    except TypeError:  # older streamlit-folium versions
        st_folium(fmap, height=620, width=1100, returned_objects=[])
