"""
Experimento 5. Comparacion con un traductor existente.
Elegir el traductor externo y registrar nombre + fecha de consulta.
Debe usar exactamente el mismo test.csv y mismas referencias.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
import pandas as pd

from src.rutas import ruta, MODELO_AJUSTADO
ruta_ajustado = MODELO_AJUSTADO
from src.metricas.calcular_metricas import calcular_bleu_chrf, guardar_resultados

NOMBRE_SISTEMA_EXTERNO = "TODO: nombre del traductor externo"
FECHA_CONSULTA = datetime.date.today().isoformat()


def main():
    test = pd.read_csv(ruta("datos_evaluacion", "test.csv"))

    # TODO: reemplazar por las traducciones reales obtenidas del sistema externo
    traducciones_externas = ["TODO" for _ in range(len(test))]

    metricas = calcular_bleu_chrf(traducciones_externas, test["ru"].tolist())
    metricas["sistema"] = NOMBRE_SISTEMA_EXTERNO
    metricas["fecha_consulta"] = FECHA_CONSULTA

    guardar_resultados(ruta("experimentos", "tablas", "experimento5.json"), metricas)
    print(metricas)


if __name__ == "__main__":
    main()
