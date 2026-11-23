"""
Calculo de BLEU y chrF con SacreBLEU, siguiendo el procedimiento obligatorio de la guia: congelar prueba, generar todas las traducciones, calcular con
las mismas referencias y configuracion de tokenizacion.
"""

import json
import sacrebleu


def calcular_bleu_chrf(traducciones: list[str], referencias: list[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(traducciones, [referencias])
    chrf = sacrebleu.corpus_chrf(traducciones, [referencias])
    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "version_sacrebleu": sacrebleu.__version__,
        "configuracion_bleu": bleu.format(),
    }


def guardar_resultados(ruta_salida: str, resultados: dict):
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
