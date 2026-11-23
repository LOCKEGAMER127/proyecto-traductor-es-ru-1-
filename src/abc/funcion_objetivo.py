"""
Funcion objetivo f(x) = 100 - chrF(x), evaluada SOLO sobre el conjunto de VALIDACION (nunca sobre prueba)

x es un vector de parametros reales del sistema: decodificacion + pesos de combinacion (ver configs/abc_search_space.json).
"""

import sacrebleu


def chrf_sobre_validacion(traducciones_generadas: list[str], referencias: list[str]) -> float:
    #chrF (0-100) calculado con sacrebleu sobre el conjunto completo, nunca con una sola oracion.
    resultado = sacrebleu.corpus_chrf(traducciones_generadas, [referencias])
    return resultado.score


def f_objetivo(traducciones_generadas: list[str], referencias: list[str]) -> float:
    #Funcion a MINIMIZAR por ABC. f(x) = 100 - chrF(x).
    return 100.0 - chrf_sobre_validacion(traducciones_generadas, referencias)


def fitness_desde_f(valor_f: float) -> float:
    # Transformacion de f(x) a fitness positivo para la ruleta de abejas observadoras (ver Guia Tecnica ABC, seccion 1).
    if valor_f >= 0:
        return 1.0 / (1.0 + valor_f)
    return 1.0 + abs(valor_f)
