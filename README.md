# Copiloto de movilidad: rutas con A*

MVP académico con dos pestañas: **Copiloto de movilidad** para rutas y evaluación A*/UCS, y **Asistente RAG** para consultar el corpus documental mediante chat. La red vial de Medellín se descargó de OpenStreetMap.

## Ejecutar

Requiere Python 3.10 o superior. Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Los archivos `Datos/nodos_medellin.csv` y `Datos/aristas_medellin.csv` deben estar presentes. La app los lee localmente, por lo que no necesita descargar OSM cada vez que inicia.

## Algoritmos y evaluación

- A* es una implementación propia en `src/routing.py`, con cola de prioridad y reconstrucción de la ruta.
- UCS/Dijkstra es la línea base: usa el mismo grafo y el mismo costo, pero con heurística cero.
- El usuario puede optimizar distancia o tiempo estimado. Para distancia se usa la longitud OSM; para tiempo se usa `length / velocidad`, con `maxspeed` cuando existe y supuestos por tipo de vía cuando falta.
- La heurística de A* es distancia geodésica en línea recta. Para tiempo se divide por la mayor velocidad de arista de la red. Esto mantiene una cota inferior admisible bajo los costos modelados.
- La interfaz informa costo de ruta, nodos expandidos y tiempo de ejecución de ambos algoritmos, igualdad del costo óptimo y reducción porcentual de expansiones.

La comparación es para pares de nodos seleccionados por el usuario y la red descargada. Para el informe, registra varios pares origen-destino (cercanos, medios y lejanos) y reporta promedios de expansiones y tiempos; no generalices a toda la ciudad con un único par.

## Datos y limitaciones

Fuente de red: OpenStreetMap, obtenida mediante OSMnx. Los tiempos son estimaciones: no hay tráfico en tiempo real. La interfaz aún no geocodifica direcciones y permite elegir nodos por identificador y coordenadas. Esta versión tampoco calcula tarifas, emisiones ni rutas reales de Metro/bus; no interpretes sus rutas como multimodales.

Documenta en la entrega la fecha de descarga y el área consultada para la red, además de cualquier supuesto de velocidad aplicado.

## RAG y agente con herramientas

El corpus de `Datos/corpus_movilidad.json` conserva fragmentos sobre Pico y Placa y Metro, con título, URL oficial y fecha de consulta. `src/rag.py` recupera los fragmentos pertinentes mediante TF-IDF y similitud coseno; es un RAG léxico local, por lo que no descarga modelos de embeddings. `src/tools.py` expone dos herramientas al agente: `search_mobility_docs` y `plan_route`. `src/agent.py` implementa un ciclo acotado de llamadas a herramientas y observaciones.

El chat usa el endpoint hosted de NVIDIA NIM, compatible con Chat Completions y llamadas a herramientas. El modelo predeterminado es `openai/gpt-oss-20b`. Genera una API key en NVIDIA Build y guárdala en el archivo `.env` de la raíz del proyecto:

```dotenv
NVIDIA_API_KEY=nvapi-TU_CLAVE
# Opcionales; estos son los valores predeterminados de la app.
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=openai/gpt-oss-20b
```

Instala las dependencias con `python -m pip install -r requirements.txt` y luego inicia la app con `streamlit run app.py`. La aplicación carga `.env` automáticamente al arrancar. No compartas ni subas ese archivo al repositorio: está excluido mediante `.gitignore`. Sin una clave NVIDIA válida, el dashboard de rutas funciona, pero el chat no puede llamar al LLM.

El corpus es una captura consultada el 11 de noviembre de 2026. Las reglas, tarifas y horarios pueden cambiar; el asistente debe mostrar las fuentes y no presentar esos datos como verificación en tiempo real. La herramienta de rutas no modela las restricciones de Pico y Placa ni calcula recorridos reales en transporte público.
