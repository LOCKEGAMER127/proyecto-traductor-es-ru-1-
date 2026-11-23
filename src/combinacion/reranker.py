"""
Combinacion log-lineal de los tres componentes (reglas, NMT, EBMT).

score(y) = w_nmt * logprob_norm(y) + w_ebmt * sim_ebmt(y) + w_reglas * bonus_reglas(y)

logprob_norm normaliza las log-probabilidades del n-best de ESTA oracion al rango [0,1] (min-max dentro del propio n-best), para que sea comparable
con sim_ebmt y bonus_reglas, que ya viven en [0,1]. Sin esta normalizacion, w_nmt domina o se anula segun la escala arbitraria del logprob y la
combinacion deja de tener sentido (causa comun de resultados raros).

Los pesos w_nmt, w_ebmt, w_reglas son parte del vector de parametros que optimiza ABC (ver configs/abc_search_space.json).
"""

from dataclasses import dataclass


@dataclass
class PesosCombinacion:
    w_nmt: float = 1.0
    w_ebmt: float = 0.0
    w_reglas: float = 0.0


def normalizar_logprobs(logprobs: list[float]) -> list[float]:
    if not logprobs:
        return []
    minimo, maximo = min(logprobs), max(logprobs)
    if maximo - minimo < 1e-9:
        return [1.0 for _ in logprobs]
    return [(lp - minimo) / (maximo - minimo) for lp in logprobs]


def puntuar_candidatos(
    candidatos_nmt: list[tuple], # [(texto, logprob), ...] del n-best del modelo
    texto_ebmt: str | None,      # traduccion recuperada por memoria (o None)
    similitud_ebmt: float,       # 0.0 si texto_ebmt es None
    diccionario_reglas: dict,
    pesos: PesosCombinacion,
) -> list[dict]:
    # Construye la lista de candidatos finales (NMT + EBMT si aplica) con su puntaje combinado, lista para elegir el mejor.
    from src.reglas.reglas_es_ru import bonus_coincidencia_reglas

    textos = [t for t, _ in candidatos_nmt]
    logprobs = [lp for _, lp in candidatos_nmt]
    logprobs_norm = normalizar_logprobs(logprobs)

    candidatos = []
    for texto, logprob_n in zip(textos, logprobs_norm):
        bonus = bonus_coincidencia_reglas(texto, diccionario_reglas)
        puntaje = (
            pesos.w_nmt * logprob_n
            + pesos.w_ebmt * 0.0   # el candidato NMT no viene de EBMT
            + pesos.w_reglas * bonus
        )
        candidatos.append({"texto": texto, "origen": "nmt", "puntaje": puntaje})

    if texto_ebmt is not None:
        bonus = bonus_coincidencia_reglas(texto_ebmt, diccionario_reglas)
        puntaje = (
            pesos.w_nmt * 0.0
            + pesos.w_ebmt * similitud_ebmt
            + pesos.w_reglas * bonus
        )
        candidatos.append({"texto": texto_ebmt, "origen": "ebmt", "puntaje": puntaje})

    return sorted(candidatos, key=lambda c: c["puntaje"], reverse=True)


def elegir_mejor_candidato(candidatos: list[dict]) -> dict:
    # Candidatos ya viene ordenado por puntuar_candidatos(); se deja esta funcion por claridad semantica en el codigo que la llama.
    return candidatos[0]
