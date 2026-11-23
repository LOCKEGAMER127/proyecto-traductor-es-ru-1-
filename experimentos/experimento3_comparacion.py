"""
Experimento 3. Comparacion antes y despues de optimizar.
Usa datos_evaluacion/test.csv UNA SOLA VEZ, con la configuracion ya seleccionada en el experimento 2. Aqui si se usa el pipeline hibrido
completo (reglas + NMT + EBMT + reranker), porque test.csv es pequeno (1000 oraciones) y solo se evalua una vez.
Deberia de tardar algunos minutos, no horas.
Salida: experimentos/tablas/experimento3.json, experimentos/registros/experimento3_traducciones.csv
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import pandas as pd
import numpy as np

from src.rutas import ruta, MODELO_AJUSTADO
ruta_ajustado = MODELO_AJUSTADO
from src.modelo.traductor import ConfigDecodificacion
from src.combinacion.pipeline import TraductorHibrido
from src.combinacion.reranker import PesosCombinacion
from src.metricas.calcular_metricas import calcular_bleu_chrf, guardar_resultados


def main():
    test = pd.read_csv(ruta("datos_evaluacion", "test.csv"))

    ruta_modelo = ruta_ajustado if os.path.isdir(ruta_ajustado) else "Helsinki-NLP/opus-mt-es-ru"
    traductor_hibrido = TraductorHibrido(
        ruta_modelo=ruta_modelo,
        ruta_train_csv=ruta("datos_evaluacion", "train.csv"),
)

    with open(ruta("configs", "decoding_base.json")) as f:
        base = json.load(f)
    with open(ruta("experimentos", "configuraciones", "config_optimizada_abc.json")) as f:
        optima = json.load(f)

    config_base = ConfigDecodificacion(
        num_beams=base["num_beams"], length_penalty=base["length_penalty"],
        repetition_penalty=base["repetition_penalty"], max_length=base["max_length"],
    )
    pesos_base = PesosCombinacion(
        w_nmt=base.get("pesos_combinacion", {}).get("w_nmt", 1.0),
        w_ebmt=base.get("pesos_combinacion", {}).get("w_ebmt", 0.0),
        w_reglas=base.get("pesos_combinacion", {}).get("w_reglas", 0.0),
    )
    config_optima = ConfigDecodificacion(
        num_beams=max(1, round(optima["num_beams"])),
        length_penalty=optima["length_penalty"],
        repetition_penalty=optima["repetition_penalty"],
        max_length=128,
    )
    pesos_optima = PesosCombinacion(
        w_nmt=optima.get("w_nmt", 1.0),
        w_ebmt=optima.get("w_ebmt", 0.0),
        w_reglas=optima.get("w_reglas", 0.0),
    )

    resultados = {}
    for nombre, config, pesos in [("base", config_base, pesos_base), ("optimizada", config_optima, pesos_optima)]:
        inicio = time.time()
        traducciones = [
            traductor_hibrido.traducir(texto, config, pesos)["traduccion"]
            for texto in test["es"].tolist()
        ]
        tiempo = time.time() - inicio
        metricas = calcular_bleu_chrf(traducciones, test["ru"].tolist())
        metricas["tiempo_inferencia_segundos"] = tiempo
        resultados[nombre] = metricas

    resultados["diferencia"] = {
        "bleu": resultados["optimizada"]["bleu"] - resultados["base"]["bleu"],
        "chrf": resultados["optimizada"]["chrf"] - resultados["base"]["chrf"],
        "tiempo_inferencia_segundos": (
            resultados["optimizada"]["tiempo_inferencia_segundos"]
            - resultados["base"]["tiempo_inferencia_segundos"]
        ),
    }

    mejores_val = []
    for i in range(1, 4):
        df_run = pd.read_csv(ruta("experimentos", "registros", f"experimento2_run{i}.csv"))
        mejores_val.append(100 - df_run["mejor_acumulado"].min())  # convertir f(x) -> chrF
    resultados["estadistica_validacion_abc"] = {
        "mejores_chrf_por_ejecucion": mejores_val,
        "promedio": float(np.mean(mejores_val)),
        "desviacion_estandar": float(np.std(mejores_val)),
    }

    guardar_resultados(ruta("experimentos", "tablas", "experimento3.json"), resultados)
    print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
