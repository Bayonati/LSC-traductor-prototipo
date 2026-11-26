"""Extrae puntos clave (cara, cuerpo y manos) con MediaPipe HolisticLandmarker.

Salida por cuadro: arreglo (P, 3) con coordenadas x, y, z normalizadas.
Las partes que no se detectan quedan en NaN (igual que en el dataset de Kaggle GISLR).

Uso:
    python extraer_puntos.py imagen.jpg
    python extraer_puntos.py video.mp4
    python extraer_puntos.py carpeta_con_imagenes/   (secuencia de cuadros, como LSC70)
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODELO = Path(__file__).parent / "modelos" / "holistic_landmarker.task"

# Cuántos puntos trae cada parte y en qué orden se guardan en la tabla.
PARTES = {
    "cara": 478,
    "cuerpo": 33,
    "mano_izq": 21,
    "mano_der": 21,
}
TOTAL_PUNTOS = sum(PARTES.values())

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp"}


def crear_detector(modo=vision.RunningMode.IMAGE):
    """Crea el detector. modo IMAGE para fotos sueltas, VIDEO para secuencias."""
    opciones = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO)),
        running_mode=modo,
    )
    return vision.HolisticLandmarker.create_from_options(opciones)


def _lista_a_arreglo(puntos, n):
    """Convierte la lista de puntos de una parte en un arreglo (n, 3).
    Si la parte no se detectó (lista vacía), devuelve NaN."""
    salida = np.full((n, 3), np.nan, dtype=np.float32)
    for i, p in enumerate(puntos[:n]):
        salida[i] = (p.x, p.y, p.z)
    return salida


def resultado_a_arreglo(resultado):
    """Junta todas las partes de un cuadro en un solo arreglo (TOTAL_PUNTOS, 3)."""
    partes = [
        _lista_a_arreglo(resultado.face_landmarks or [], PARTES["cara"]),
        _lista_a_arreglo(resultado.pose_landmarks or [], PARTES["cuerpo"]),
        _lista_a_arreglo(resultado.left_hand_landmarks or [], PARTES["mano_izq"]),
        _lista_a_arreglo(resultado.right_hand_landmarks or [], PARTES["mano_der"]),
    ]
    return np.concatenate(partes, axis=0)


def _a_imagen_mp(cuadro_bgr):
    """OpenCV lee en BGR; MediaPipe espera RGB."""
    rgb = cv2.cvtColor(cuadro_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def extraer_cuadros(cuadros_bgr, fps=30):
    """Recibe una lista de cuadros (imágenes BGR) y devuelve (T, TOTAL_PUNTOS, 3).
    Usa el modo VIDEO, que aprovecha el cuadro anterior para seguir mejor los puntos."""
    if len(cuadros_bgr) == 0:
        return np.empty((0, TOTAL_PUNTOS, 3), dtype=np.float32)
    salida = []
    with crear_detector(vision.RunningMode.VIDEO) as det:
        for i, cuadro in enumerate(cuadros_bgr):
            marca_ms = int(i * 1000 / fps)  # el modo VIDEO exige tiempos crecientes
            res = det.detect_for_video(_a_imagen_mp(cuadro), marca_ms)
            salida.append(resultado_a_arreglo(res))
    return np.stack(salida)


def leer_entrada(ruta):
    """Devuelve (lista de cuadros BGR, fps) desde una imagen, un video o una carpeta de imágenes."""
    ruta = Path(ruta)
    if ruta.is_dir():
        archivos = sorted(p for p in ruta.iterdir() if p.suffix.lower() in EXT_IMAGEN)
        return [cv2.imread(str(p)) for p in archivos], 30
    if ruta.suffix.lower() in EXT_IMAGEN:
        img = cv2.imread(str(ruta))
        if img is None:
            raise FileNotFoundError(f"No pude abrir la imagen: {ruta}")
        return [img], 30
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir el video: {ruta}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cuadros = []
    while True:
        ok, cuadro = cap.read()
        if not ok:
            break
        cuadros.append(cuadro)
    cap.release()
    return cuadros, fps


def resumen(arr):
    """Cuenta en cuántos cuadros se detectó cada parte."""
    inicio = 0
    texto = []
    for nombre, n in PARTES.items():
        bloque = arr[:, inicio:inicio + n, 0]
        detectados = int((~np.isnan(bloque).all(axis=1)).sum())
        texto.append(f"  {nombre:9s}: detectada en {detectados}/{len(arr)} cuadros")
        inicio += n
    return "\n".join(texto)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    cuadros, fps = leer_entrada(sys.argv[1])
    puntos = extraer_cuadros(cuadros, fps)
    print(f"Forma del resultado: {puntos.shape}  (cuadros, puntos, xyz)")
    print(resumen(puntos))
    destino = Path(sys.argv[1]).with_suffix(".npy") if not Path(sys.argv[1]).is_dir() \
        else Path(sys.argv[1]).parent / (Path(sys.argv[1]).name + ".npy")
    np.save(destino, puntos)
    print(f"Guardado en: {destino}")
