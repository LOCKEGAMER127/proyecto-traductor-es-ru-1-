"""
Resuelve rutas absolutas desde la raiz del repositorio, para que los scripts
de experimentos/ funcionen sin importar desde que carpeta se invoquen.
"""
import os

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELO_AJUSTADO = r"C:/modelos/opus-mt-es-ru-ajustado"


def ruta(*partes) -> str:
    return os.path.join(RAIZ, *partes)