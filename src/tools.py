"""Herramientas que puede invocar el agente de movilidad."""

from __future__ import annotations

import json

from src.routing import compare_search

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "plan_route",
            "description": "Calcula una ruta vial estimada entre dos nodos de la red local de Medellín con A* y compara su costo con UCS. Usa los IDs numéricos seleccionados por el usuario en la interfaz; si el usuario indica otros IDs válidos, usa esos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "integer", "description": "ID OSM del nodo origen seleccionado en la interfaz."},
                    "destination": {"type": "integer", "description": "ID OSM del nodo destino seleccionado en la interfaz."},
                    "metric": {"type": "string", "enum": ["distance", "time"], "description": "Optimizar por distancia o tiempo estimado."},
                },
                "required": ["origin", "destination", "metric"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_mobility_docs",
            "description": "Busca evidencia en el corpus documental sobre pico y placa, vías exentas, Metro, horarios, tarifas o bicicletas. Usa esta herramienta para responder datos normativos u operacionales y cita las fuentes devueltas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Pregunta o conceptos que se deben buscar en los documentos."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(name: str, arguments: dict, *, graph, coords, retriever, default_origin: int, default_destination: int, default_metric: str):
    if name == "search_mobility_docs":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"error": "La búsqueda requiere una pregunta."}
        found = retriever.search(query, top_k=3)
        if not found:
            return {"message": "No encontré evidencia suficiente en el corpus local.", "sources": []}
        return {
            "results": [
                {
                    "title": item["title"],
                    "source": item["source"],
                    "url": item["url"],
                    "consulted_at": item["consulted_at"],
                    "similarity": item["score"],
                    "evidence": item["text"],
                }
                for item in found
            ]
        }

    if name == "plan_route":
        origin = int(arguments.get("origin", default_origin))
        destination = int(arguments.get("destination", default_destination))
        metric = str(arguments.get("metric", default_metric))
        if origin not in graph or destination not in graph:
            return {"error": "El ID indicado no está en la red. Usa los IDs de origen y destino mostrados en la interfaz."}
        if metric not in ("distance", "time"):
            return {"error": "El criterio debe ser distance o time."}
        astar, ucs, comparison = compare_search(graph, coords, origin, destination, metric)
        if astar is None:
            return {"error": f"No se encontró una ruta dirigida entre {origin} y {destination}."}
        return {
            "origin": origin,
            "destination": destination,
            "metric": metric,
            "distance_km": round(astar["distance_m"] / 1000, 3),
            "estimated_minutes": round(astar["time_s"] / 60, 1),
            "optimized_cost": round(astar["objective_cost"], 3),
            "route_nodes": astar["path"],
            "road_names": list(dict.fromkeys(
                edge.street_name for edge in astar["edges"] if edge.street_name
            )),
            "route_node_count": len(astar["path"]),
            "astar_expanded": astar["expanded"],
            "ucs_expanded": ucs["expanded"],
            "same_optimal_cost": comparison["same_optimal_cost"],
            "astar_expansions_saved": comparison["expansions_saved"],
            "limitations": "Estimación en red vial drive; no representa tráfico en tiempo real ni una ruta de Metro/bus.",
        }

    return {"error": f"Herramienta desconocida: {name}"}


def execute_tool_json(name: str, raw_arguments: str, **context) -> str:
    try:
        arguments = json.loads(raw_arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("Los argumentos deben ser un objeto JSON.")
        result = execute_tool(name, arguments, **context)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        result = {"error": f"Argumentos inválidos para {name}: {exc}"}
    except Exception as exc:
        result = {"error": f"La herramienta {name} falló: {exc}"}
    return json.dumps(result, ensure_ascii=False)
