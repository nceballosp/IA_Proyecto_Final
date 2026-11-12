"""Loop acotado de chat completions con llamadas a herramientas."""

from __future__ import annotations

import requests

from src.tools import TOOL_SPECS, execute_tool_json

SYSTEM_PROMPT = """Eres el copiloto de movilidad del Valle de Aburrá. Responde en español, de forma clara y breve.

REGLAS:
- Para preguntas factuales sobre pico y placa, vías exentas, Metro, horarios, tarifas o bicicletas, llama search_mobility_docs antes de responder.
- Para calcular o explicar una ruta, llama plan_route. Los nodos disponibles y criterio preferido están en el contexto de esta conversación. No inventes direcciones ni IDs.
- Puedes encadenar ambas herramientas si la pregunta lo requiere.
- Cita cada dato documental con el título y el enlace exacto devuelto por la herramienta, por ejemplo [Metro de Medellín - Usuarios](URL). Indica la fecha de consulta cuando sea relevante.
- Si la recuperación no encuentra evidencia, dilo con claridad; no inventes normas, tarifas ni horarios.
- La herramienta de ruta solo modela calles para conducir y tiempos estimados; no calcula rutas reales de Metro o bus, tráfico actual, tarifas ni emisiones.
- No afirmes que se aplicaron restricciones o vías exentas al algoritmo de ruta: la búsqueda de calles aún no modela Pico y Placa.
- Si el usuario pregunta por una ruta pero no da origen/destino, usa los nodos seleccionados en el contexto; si no están disponibles, pide que los elija en el panel.
"""


def _chat_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def answer_with_tools(
    *,
    user_text: str,
    history: list[dict],
    base_url: str,
    api_key: str,
    model: str,
    graph,
    coords,
    retriever,
    default_origin: int,
    default_destination: int,
    default_metric: str,
    max_tool_rounds: int = 4,
    timeout_seconds: int = 60,
) -> str:
    if not api_key:
        raise ValueError("Configura NVIDIA_API_KEY antes de usar el chat.")
    if not base_url or not model:
        raise ValueError("Configura NVIDIA_BASE_URL y NVIDIA_MODEL para activar el agente.")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    context = (
        f"Nodos seleccionados actualmente: origen={default_origin}, destino={default_destination}. "
        f"Criterio seleccionado: {default_metric}. La red es local y está disponible con la herramienta plan_route."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCONTEXTO DE SESIÓN: " + context}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_text})
    tool_context = {
        "graph": graph,
        "coords": coords,
        "retriever": retriever,
        "default_origin": default_origin,
        "default_destination": default_destination,
        "default_metric": default_metric,
    }

    for _ in range(max_tool_rounds):
        response = requests.post(
            _chat_url(base_url),
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "tools": TOOL_SPECS,
                "tool_choice": "auto",
                "temperature": 0.2,
            },
            timeout=timeout_seconds,
        )
        if not response.ok:
            detail = response.text[:500]
            raise RuntimeError(f"El proveedor del LLM respondió HTTP {response.status_code}: {detail}")
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("El proveedor devolvió una respuesta sin choices.")
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return (message.get("content") or "No se recibió texto en la respuesta del agente.").strip()

        messages.append(message)
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            observation = execute_tool_json(name, function.get("arguments", "{}"), **tool_context)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", name),
                "name": name,
                "content": observation,
            })
    return "No pude completar el ciclo de herramientas dentro del límite de pasos. Reformula la pregunta o consulta la fuente directamente."
