"""
Experimento 4. Comparacion cualitativa de traducciones.
Selecciona 5 ejemplos de test.csv con distintas longitudes y compara texto original, referencia, traduccion base y traduccion optimizada.
Completar 'diferencias_observadas' manualmente (omisiones, orden, concordancia, errores de significado).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd

from src.rutas import ruta, MODELO_AJUSTADO
ruta_ajustado = MODELO_AJUSTADO
from src.modelo.traductor import ConfigDecodificacion
from src.combinacion.pipeline import TraductorHibrido
from src.combinacion.reranker import PesosCombinacion


def main():
    test = pd.read_csv(ruta("datos_evaluacion", "test.csv"))
    test["longitud"] = test["es"].str.split().apply(len)
    indices = sorted(set([0, len(test) // 4, len(test) // 2, (3 * len(test)) // 4, len(test) - 1]))
    muestra = test.sort_values("longitud").iloc[indices]

    muestra = test.sort_values("longitud").iloc[indices]

    ruta_modelo = ruta_ajustado if os.path.isdir(ruta_ajustado) else "Helsinki-NLP/opus-mt-es-ru"
    traductor_hibrido = TraductorHibrido(
        ruta_modelo=ruta_modelo, ruta_train_csv=ruta("datos_evaluacion", "train.csv")
    )

    with open(ruta("configs", "decoding_base.json")) as f:
        base = json.load(f)
    with open(ruta("experimentos", "configuraciones", "config_optimizada_abc.json")) as f:
        optima = json.load(f)

    config_base = ConfigDecodificacion(
        num_beams=base["num_beams"], length_penalty=base["length_penalty"],
        repetition_penalty=base["repetition_penalty"], max_length=base["max_length"],
    )
    config_optima = ConfigDecodificacion(
        num_beams=max(1, round(optima["num_beams"])), length_penalty=optima["length_penalty"],
        repetition_penalty=optima["repetition_penalty"], max_length=128,
    )

    filas = []
    for _, fila in muestra.iterrows():
        filas.append({
            "texto_original": fila["es"],
            "referencia": fila["ru"],
            "traduccion_base": traductor_hibrido.traducir(fila["es"], config_base, PesosCombinacion())["traduccion"],
            "traduccion_optimizada": traductor_hibrido.traducir(fila["es"], config_optima, PesosCombinacion())["traduccion"],
            "diferencias_observadas": "TODO: completar tras revision manual",
        })

    pd.DataFrame(filas).to_csv(ruta("experimentos", "tablas", "experimento4_ejemplos.csv"), index=False)
    print("Ejemplos guardados en experimentos/tablas/experimento4_ejemplos.csv")


if __name__ == "__main__":
    main()
