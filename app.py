"""Dashboard con una pestaña de rutas y otra de asistente RAG."""

from pathlib import Path
import os
import sys

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.agent import answer_with_tools  # noqa: E402
from src.rag import MobilityRetriever  # noqa: E402
from src.routing import build_node_labels, compare_search, load_network  # noqa: E402


st.set_page_config(page_title="Copiloto de movilidad | Medellín",
                   page_icon="🚌", layout="wide")


@st.cache_resource(show_spinner="Cargando la red vial local...")
def get_network():
    return load_network(ROOT / "Datos")


@st.cache_resource(show_spinner="Preparando el corpus de movilidad...")
def get_retriever():
    return MobilityRetriever(ROOT / "Datos" / "corpus_movilidad.json")


st.title("Copiloto de movilidad · Medellín")
st.caption("Explora rutas sobre la red vial local y consulta un asistente con recuperación de fuentes oficiales.")

try:
    nodes, graph, coords = get_network()
except Exception as exc:
    st.error(f"No pude cargar los datos locales: {exc}")
    st.info(
        "Verifica que existan Datos/nodos_medellin.csv y Datos/aristas_medellin.csv.")
    st.stop()

node_ids = list(coords)
node_labels = build_node_labels(graph, coords)
center_lat = sum(lat for lat, _ in coords.values()) / len(coords)
center_lon = sum(lon for _, lon in coords.values()) / len(coords)

tab_routes, tab_rag = st.tabs(["Copiloto de movilidad", "Asistente RAG"])

with tab_routes:
    st.header("Planifica y compara rutas")
    st.caption("A* y UCS/Dijkstra usan la misma red y el mismo criterio de costo.")
    controls = st.columns([1, 1, 1, 1])
    metric_label = controls[0].radio(
        "Optimizar por", ["Distancia", "Tiempo estimado"])
    metric = "distance" if metric_label == "Distancia" else "time"
    origin = controls[1].selectbox(
        "Origen (busca por calle)", node_ids, index=0,
        format_func=lambda n: node_labels[n], key="route_origin",
    )
    destination_index = min(1, len(node_ids) - 1)
    destination = controls[2].selectbox(
        "Destino (busca por calle)", node_ids, index=destination_index,
        format_func=lambda n: node_labels[n], key="route_destination",
    )
    run_route = controls[3].button(
        "Calcular ruta", type="primary", width='stretch')
    st.caption("Los datos no contienen direcciones geocodificadas. Los tiempos son estimaciones y no representan tráfico en tiempo real.")

    if "route_result" not in st.session_state:
        st.session_state.route_result = None

    if run_route:
        if origin == destination:
            st.warning("Elige nodos de origen y destino distintos.")
        else:
            with st.spinner("Buscando rutas y midiendo expansiones..."):
                astar, ucs, comparison = compare_search(
                    graph, coords, origin, destination, metric)
            st.session_state.route_result = (
                astar, ucs, comparison, metric, origin, destination)

    saved = st.session_state.route_result
    if saved is None:
        st.info("Selecciona origen y destino y pulsa **Calcular ruta**.")
        st.map(pd.DataFrame(
            {"lat": [center_lat], "lon": [center_lon]}), zoom=11)
    else:
        astar, ucs, comparison, saved_metric, saved_origin, saved_destination = saved
        if astar is None:
            st.error(
                "No existe una ruta dirigida entre esos nodos en la red descargada.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Distancia A*", f"{astar['distance_m'] / 1000:.2f} km")
            col2.metric("Tiempo estimado A*",
                        f"{astar['time_s'] / 60:.1f} min")
            col3.metric("Nodos expandidos · A*", f"{astar['expanded']:,}")
            col4.metric(
                "Nodos expandidos · UCS", f"{ucs['expanded']:,}",
                delta=f"{comparison['expansions_saved']:+,} vs A*",
            )

            left, right = st.columns([1.55, 1])
            with left:
                st.subheader("Ruta sobre el mapa")
                route_points = []
                for edge in astar["edges"]:
                    route_points.extend((lon, lat)
                                        for lat, lon in edge.geometry)
                if not route_points:
                    route_points = [(coords[n][1], coords[n][0])
                                    for n in astar["path"]]
                path_layer = pdk.Layer(
                    "PathLayer", data=[{"path": route_points}], get_path="path",
                    get_color=[30, 120, 220], width_scale=12, width_min_pixels=4, pickable=True,
                )
                marker_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame([
                        {"lon": coords[saved_origin][1],
                            "lat": coords[saved_origin][0], "label": "Origen"},
                        {"lon": coords[saved_destination][1],
                            "lat": coords[saved_destination][0], "label": "Destino"},
                    ]),
                    get_position="[lon, lat]", get_fill_color=[220, 60, 60],
                    get_radius=55, radius_min_pixels=6, pickable=True,
                )
                view = pdk.ViewState(
                    latitude=coords[saved_origin][0], longitude=coords[saved_origin][1], zoom=12, pitch=0,
                )
                st.pydeck_chart(
                    pdk.Deck(layers=[path_layer, marker_layer], initial_view_state=view,
                             tooltip={"text": "{label}"}),
                    width='stretch',
                )

            with right:
                st.subheader("Evaluación del algoritmo")
                criterion = "distancia (m)" if saved_metric == "distance" else "tiempo estimado (s)"
                st.markdown(
                    f"**Criterio:** {criterion}  \n"
                    f"**Mismo costo óptimo:** {'Sí' if comparison['same_optimal_cost'] else 'No'}"
                )
                st.caption(
                    "UCS usa h(n)=0. A* usa distancia geodésica como cota inferior; para tiempo la divide por la mayor velocidad de la red.")
                st.bar_chart(pd.DataFrame(
                    {"Nodos expandidos": [astar["expanded"], ucs["expanded"]]},
                    index=["A*", "UCS / Dijkstra"],
                ))
                st.dataframe(pd.DataFrame([
                    {"Algoritmo": "A*", "Expansiones": astar["expanded"],
                     "Tiempo (ms)": round(astar["elapsed_ms"], 2),
                     "Costo objetivo": round(astar["objective_cost"], 2)},
                    {"Algoritmo": "UCS / Dijkstra", "Expansiones": ucs["expanded"],
                     "Tiempo (ms)": round(ucs["elapsed_ms"], 2),
                     "Costo objetivo": round(ucs["objective_cost"], 2)},
                ]), hide_index=True, width='stretch')
                st.metric("Reducción de expansiones con A*",
                          f"{comparison['expansion_reduction_pct']:.1f}%")
                st.write(
                    f"**Origen:** {node_labels[saved_origin]}  \n"
                    f"**Destino:** {node_labels[saved_destination]}  \n"
                    f"La ruta atraviesa {len(astar['path'])} nodos. La velocidad se toma de OSM cuando existe y se estima por tipo de vía cuando falta."
                )

with tab_rag:
    st.header("Asistente documental de movilidad")
    st.caption(
        "Responde sobre Pico y Placa e información del Metro como horarios, rutas integradas, etc. Cita las fuentes consultadas.")
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    for message in st.session_state.agent_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Pregunta por Pico y Placa, Horarios del Metro, Rutas Integradas")
    if prompt:
        prior_history = list(st.session_state.agent_history)
        st.session_state.agent_history.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response_text = answer_with_tools(
                    user_text=prompt,
                    history=prior_history,
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    model="meta/llama-3.2-90b-vision-instruct",
                    graph=graph,
                    coords=coords,
                    retriever=get_retriever(),
                    default_origin=origin,
                    default_destination=destination,
                    default_metric=metric,
                )
            except Exception as exc:
                response_text = f"No pude completar la consulta: {exc}"
            st.markdown(response_text)
        st.session_state.agent_history.append(
            {"role": "assistant", "content": response_text})
        st.rerun()
