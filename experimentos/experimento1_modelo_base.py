"""
Entrada: datos_evaluacion/test.csv (ya congelado, para que ni  le muevan).
Salida: experimentos/tablas/experimento1.json, experimentos/registros/experimento1_traducciones.csv
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import pandas as pd

from src.rutas import ruta, MODELO_AJUSTADO
ruta_ajustado = MODELO_AJUSTADO
from src.modelo.traductor import TraductorEsRu, ConfigDecodificacion
from src.metricas.calcular_metricas import calcular_bleu_chrf, guardar_resultados


def main():
    test = pd.read_csv(ruta("datos_evaluacion", "test.csv"))
    with open(ruta("configs", "decoding_base.json")) as f:
        datos_config = json.load(f)
    config = ConfigDecodificacion(
        num_beams=datos_config["num_beams"],
        length_penalty=datos_config["length_penalty"],
        repetition_penalty=datos_config["repetition_penalty"],
        max_length=datos_config["max_length"],
    )

    ruta_modelo = ruta_ajustado if os.path.isdir(ruta_ajustado) else "Helsinki-NLP/opus-mt-es-ru"
    print(f"Usando modelo: {ruta_modelo}")
    traductor = TraductorEsRu(ruta_modelo=ruta_modelo)

    inicio = time.time()
    traducciones = traductor.traducir_lote(test["es"].tolist(), config)
    tiempo_total = time.time() - inicio

    assert len(traducciones) == len(test), "Desalineacion hipotesis/referencia: revisar traducir_lote."

    metricas = calcular_bleu_chrf(traducciones, test["ru"].tolist())
    metricas["tiempo_inferencia_segundos"] = tiempo_total
    metricas["n_oraciones"] = len(test)
    metricas["modelo_usado"] = ruta_modelo

    guardar_resultados(ruta("experimentos", "tablas", "experimento1.json"), metricas)
    pd.DataFrame({
        "es": test["es"], "referencia": test["ru"], "traduccion_base": traducciones
    }).to_csv(ruta("experimentos", "registros", "experimento1_traducciones.csv"), index=False)

    print(json.dumps(metricas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
