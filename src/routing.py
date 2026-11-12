"""Carga de la red OSM y búsqueda A*/UCS para el copiloto de movilidad."""

from __future__ import annotations

import heapq
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datos"
Metric = Literal["distance", "time"]


@dataclass(frozen=True)
class Edge:
    target: int
    length_m: float
    seconds: float
    geometry: tuple[tuple[float, float], ...]
    street_name: str = ""


def _parse_speed(value: object, highway: object) -> float:
    """Velocidad en km/h; usa un supuesto conservador si OSM no la informa."""
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match:
        speed = float(match.group())
        if "mph" in str(value).lower():
            speed *= 1.60934
        return min(max(speed, 5.0), 120.0)

    road = str(highway).lower()
    defaults = {
        "motorway": 80, "trunk": 70, "primary": 50,
        "secondary": 40, "tertiary": 30, "residential": 25,
        "living_street": 10, "service": 15,
    }
    for kind, speed in defaults.items():
        if kind in road:
            return float(speed)
    return 25.0


def _geometry_points(value: object, start: tuple[float, float], end: tuple[float, float]):
    """Lee coordenadas de LINESTRING WKT y las devuelve como (lat, lon)."""
    matches = re.findall(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", str(value))
    if not matches:
        return (start, end)
    return tuple((float(lat), float(lon)) for lon, lat in matches)


def load_network(data_dir: Path = DATA_DIR):
    nodes = pd.read_csv(data_dir / "nodos_medellin.csv")
    edges = pd.read_csv(data_dir / "aristas_medellin.csv")
    nodes["osmid"] = pd.to_numeric(nodes["osmid"], errors="raise").astype("int64")
    nodes["y"] = pd.to_numeric(nodes["y"], errors="raise")
    nodes["x"] = pd.to_numeric(nodes["x"], errors="raise")
    edges["u"] = pd.to_numeric(edges["u"], errors="raise").astype("int64")
    edges["v"] = pd.to_numeric(edges["v"], errors="raise").astype("int64")
    edges["length"] = pd.to_numeric(edges["length"], errors="coerce")
    nodes = nodes.set_index("osmid", drop=False)
    coords = {int(row.osmid): (float(row.y), float(row.x)) for row in nodes.itertuples()}

    graph: dict[int, list[Edge]] = {node_id: [] for node_id in coords}
    valid = edges.dropna(subset=["length"])
    for row in valid.itertuples(index=False):
        u, v, length = int(row.u), int(row.v), float(row.length)
        if u not in coords or v not in coords or length < 0 or not math.isfinite(length):
            continue
        speed = _parse_speed(getattr(row, "maxspeed", None), getattr(row, "highway", None))
        geom = _geometry_points(getattr(row, "geometry", ""), coords[u], coords[v])
        raw_name = getattr(row, "name", None)
        street_name = "" if pd.isna(raw_name) else str(raw_name).strip()
        graph[u].append(Edge(v, length, length / (speed / 3.6), geom, street_name))
    return nodes, graph, coords


def build_node_labels(graph, coords: dict[int, tuple[float, float]]) -> dict[int, str]:
    """Crea etiquetas de nodos usando los nombres de las calles conectadas."""
    streets_by_node: dict[int, set[str]] = {node_id: set() for node_id in coords}
    for source, edges in graph.items():
        for edge in edges:
            if edge.street_name:
                streets_by_node[source].add(edge.street_name)
                streets_by_node[edge.target].add(edge.street_name)

    labels = {}
    for node_id, streets in streets_by_node.items():
        lat, lon = coords[node_id]
        if streets:
            street_label = " · ".join(sorted(streets)[:2])
            labels[node_id] = f"{street_label} · nodo {node_id}"
        else:
            labels[node_id] = f"Cruce sin nombre · nodo {node_id} · {lat:.4f}, {lon:.4f}"
    return labels


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def nearest_node(coords: dict[int, tuple[float, float]], lat: float, lon: float) -> int:
    return min(coords, key=lambda n: haversine_m(coords[n], (lat, lon)))


def search(graph, coords, start: int, goal: int, metric: Metric = "distance", algorithm: str = "astar"):
    """Búsqueda A* (heurística admisible) o UCS/Dijkstra; cuenta expansiones y tiempo."""
    if start not in graph or goal not in graph:
        raise ValueError("El nodo de origen o destino no pertenece a la red.")
    if metric not in ("distance", "time"):
        raise ValueError("La métrica debe ser 'distance' o 'time'.")
    max_speed_mps = max((edge.length_m / edge.seconds for adj in graph.values() for edge in adj if edge.seconds), default=1)

    def edge_cost(edge: Edge) -> float:
        return edge.length_m if metric == "distance" else edge.seconds

    def heuristic(node: int) -> float:
        if algorithm != "astar":
            return 0.0
        straight = haversine_m(coords[node], coords[goal])
        return straight if metric == "distance" else straight / max_speed_mps

    start_time = time.perf_counter()
    frontier = [(heuristic(start), 0.0, start)]
    costs = {start: 0.0}
    parent: dict[int, int] = {}
    expanded = 0
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if cost != costs.get(current):
            continue
        expanded += 1
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            route_edges = []
            for u, v in zip(path, path[1:]):
                route_edges.append(min((e for e in graph[u] if e.target == v), key=edge_cost))
            return {
                "path": path,
                "distance_m": sum(e.length_m for e in route_edges),
                "time_s": sum(e.seconds for e in route_edges),
                "objective_cost": cost,
                "expanded": expanded,
                "elapsed_ms": (time.perf_counter() - start_time) * 1000,
                "edges": route_edges,
            }
        for edge in graph[current]:
            candidate = cost + edge_cost(edge)
            if candidate < costs.get(edge.target, math.inf):
                costs[edge.target] = candidate
                parent[edge.target] = current
                heapq.heappush(frontier, (candidate + heuristic(edge.target), candidate, edge.target))
    return None


def compare_search(graph, coords, start: int, goal: int, metric: Metric = "distance"):
    """Compara A* y UCS con misma métrica y red; retorna resultados y métricas."""
    astar = search(graph, coords, start, goal, metric, "astar")
    ucs = search(graph, coords, start, goal, metric, "ucs")
    if astar is None or ucs is None:
        return astar, ucs, None
    comparison = {
        "same_optimal_cost": math.isclose(astar["objective_cost"], ucs["objective_cost"], rel_tol=1e-9, abs_tol=1e-6),
        "expansions_saved": ucs["expanded"] - astar["expanded"],
        "expansion_reduction_pct": 100 * (ucs["expanded"] - astar["expanded"]) / max(ucs["expanded"], 1),
        "time_ratio": astar["elapsed_ms"] / max(ucs["elapsed_ms"], 1e-9),
    }
    return astar, ucs, comparison
