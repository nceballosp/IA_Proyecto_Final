# Copiloto de movilidad: rutas con A*

MVP académico para calcular rutas sobre la red vial de Medellín descargada de OpenStreetMap. La interfaz compara A* con búsqueda de costo uniforme (UCS/Dijkstra) y muestra la ruta en un mapa.

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
