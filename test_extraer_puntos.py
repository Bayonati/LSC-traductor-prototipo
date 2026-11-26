"""Pruebas de extraer_puntos.py. Ejecutar: python test_extraer_puntos.py"""

from pathlib import Path

import numpy as np

from extraer_puntos import (PARTES, TOTAL_PUNTOS, extraer_cuadros, leer_entrada,
                            _lista_a_arreglo)

AQUI = Path(__file__).parent


class P:  # punto falso para probar la conversión sin MediaPipe
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


# 1. Entrada vacía: cero cuadros -> arreglo vacío con la forma correcta
vacio = extraer_cuadros([])
assert vacio.shape == (0, TOTAL_PUNTOS, 3)

# 2. Parte no detectada (lista vacía) -> todo NaN
assert np.isnan(_lista_a_arreglo([], 21)).all()

# 3. Un solo punto: se copia en la fila 0 y el resto queda NaN
uno = _lista_a_arreglo([P(0.1, 0.2, -0.3)], 21)
assert np.allclose(uno[0], [0.1, 0.2, -0.3])  # z negativo = más cerca de la cámara
assert np.isnan(uno[1:]).all()

# 4. Puntos duplicados (iguales) se guardan tal cual, sin fusionarse
dup = _lista_a_arreglo([P(0.5, 0.5, 0.0)] * 3, 3)
assert np.allclose(dup, 0.5 * np.array([[1, 1, 0]] * 3))

# 5. Imagen en negro: no hay persona -> todas las partes en NaN
negro = np.zeros((480, 640, 3), dtype=np.uint8)
r = extraer_cuadros([negro])
assert r.shape == (1, TOTAL_PUNTOS, 3)
assert np.isnan(r).all()

# 6. Foto real (brazos cruzados, mano tapando la cara): el cuerpo sí se detecta
#    y sus coordenadas x, y quedan cerca de [0, 1].
#    Ojo: en esta foto Holistic NO encuentra las manos ni la cara (limitación real).
FOTO = AQUI / "muestras" / "manos.jpg"
if FOTO.exists():
    cuadros, fps = leer_entrada(FOTO)
    r = extraer_cuadros(cuadros, fps)
    ini, fin = PARTES["cara"], PARTES["cara"] + PARTES["cuerpo"]
    cuerpo = r[0, ini:fin]
    assert not np.isnan(cuerpo).any(), "No detectó el cuerpo en la foto de ejemplo"
    assert ((cuerpo[:, :2] > -0.5) & (cuerpo[:, :2] < 1.5)).all()
else:
    print("Prueba 6 omitida: falta muestras/manos.jpg")

print("Todas las pruebas pasaron.")
