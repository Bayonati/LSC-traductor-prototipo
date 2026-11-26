# Traductor de Lengua de Señas Colombiana (LSC) – Prototipo

Proyecto Dirigido – Pontificia Universidad Javeriana Cali.

**Alcances y limitaciones del aprendizaje automático para el reconocimiento de Lengua de Señas Colombiana (LSC) a texto en español.**

>  En construcción. Próximamente: prueba de concepto que reconoce 10 palabras en LSC a partir de imágenes del dataset LSC70, resultados y video pitch.

## Qué hay hasta ahora

| Archivo | Qué hace |
|---|---|
| `extraer_puntos.py` | Extrae 553 puntos clave (cara, cuerpo y manos) por cuadro con MediaPipe Holistic |
| `test_extraer_puntos.py` | Pruebas automáticas de la extracción |
| `docs/1_Marco_teorico_y_estado_del_arte.pdf` | Problemática, variables, árbol de problemas y estado del arte |
| `docs/2_Plan_de_trabajo.pdf` | Enfoque, experimentos, evaluación y avance logrado |

## Instalación (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install mediapipe opencv-python numpy
New-Item -ItemType Directory -Force modelos
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task' -OutFile modelos\holistic_landmarker.task
.\.venv\Scripts\python.exe test_extraer_puntos.py
```

## Datos

Se usa **LSC70** (Universidad del Cauca, licencia CC BY 4.0): https://doi.org/10.17632/9ssyn8tff5.2
Descomprimir en `datos/LSC70/`. El dataset no se incluye en el repositorio por su tamaño (~1 GB).
