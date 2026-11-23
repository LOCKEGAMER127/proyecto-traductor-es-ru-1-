"""
Capa de reglas (RBMT ligero) para el traductor es-ru. No reemplaza al modelo: lo condiciona antes (pre-procesamiento) y despues (post-proceso
de terminologia / tipografia).

Reglas implementadas (documentar en el articulo, seccion Metodologia):
R1. Normalizacion de espacios y NBSP.
R2. Normalizacion de comillas: comillas rectas/inglesas "..." -> comillas
    angulares rusas «...» (convencion tipografica del ruso).
R3. Signos de apertura en espanol (¿ ¡) se eliminan antes de traducir
    (el ruso no los usa) para no confundir al tokenizador.
R4. Colapso de puntuacion repetida ("..!!" -> ".").
R5. Diccionario forzado de terminologia (nombres propios, siglas).
"""

import json
import re
import os

RUTA_DICCIONARIO = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "diccionario_forzado_es_ru.json"
)

_PATRON_ESPACIOS = re.compile(r"\s+")
_PATRON_PUNTUACION_REPETIDA = re.compile(r"([.!?,])\1+")
_PATRON_APERTURA_ES = re.compile(r"[¿¡]")


def cargar_diccionario_forzado(ruta: str = RUTA_DICCIONARIO) -> dict:
    """
    Carga pares termino_es -> termino_ru que deben traducirse siempre igual (nombres propios, siglas, terminologia tecnica). Ignora la
    llave '_comentario' si existe.
    """
    if not os.path.exists(ruta):
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return {k: v for k, v in datos.items() if not k.startswith("_")}


def normalizar_entrada(texto: str) -> str:
    # R1 + R3 + R4: normalizacion previa a la traduccion.
    texto = texto.replace("\xa0", " ")
    texto = _PATRON_APERTURA_ES.sub("", texto)
    texto = _PATRON_PUNTUACION_REPETIDA.sub(r"\1", texto)
    texto = _PATRON_ESPACIOS.sub(" ", texto).strip()
    if texto and texto[0].islower():
        texto = texto[0].upper() + texto[1:]
    return texto


def normalizar_tipografia_rusa(texto: str) -> str:
    """
    R2: convierte comillas rectas/inglesas a comillas angulares rusas.
    Aplicar despues de traducir, sobre la salida en ruso.
    Pares de comillas dobles "..." -> «...»
    """
    texto = re.sub(r'"([^"]*)"', r"«\1»", texto)
    return texto


def aplicar_terminologia_forzada(texto_traducido: str, diccionario: dict) -> str:
    """
    R5: sustituye en la salida los terminos que deben respetar el diccionario forzado. Usa limites de palabra e insensibilidad a mayusculas
    sobre el termino en espanol que dispara la sustitucion.
    """
    resultado = texto_traducido
    for termino_es, termino_ru in diccionario.items():
        patron = re.compile(re.escape(termino_es), re.IGNORECASE)
        if patron.search(resultado):
            resultado = patron.sub(termino_ru, resultado)
    return resultado


def bonus_coincidencia_reglas(texto_candidato: str, diccionario: dict) -> float:
    # Puntaje usado por el reranker (0-1): fraccion de terminos forzados que aparecen correctamente en el candidato
    if not diccionario:
        return 0.0
    aciertos = sum(1 for term_ru in diccionario.values() if term_ru in texto_candidato)
    return aciertos / max(1, len(diccionario))


def pipeline_reglas_pre(texto: str) -> str:
    # Punto de entrada unico para el pre-procesamiento (usar antes de pasar el texto al modelo).
    return normalizar_entrada(texto)


def pipeline_reglas_post(texto_traducido: str, diccionario: dict = None) -> str:
    # Punto de entrada unico para el post-procesamiento (usar sobre la salida del modelo, antes de mostrarla al usuario).
    diccionario = diccionario if diccionario is not None else cargar_diccionario_forzado()
    texto_traducido = normalizar_tipografia_rusa(texto_traducido)
    texto_traducido = aplicar_terminologia_forzada(texto_traducido, diccionario)
    return texto_traducido
