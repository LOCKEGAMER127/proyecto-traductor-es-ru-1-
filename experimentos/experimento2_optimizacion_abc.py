"""
Optimizacion bioinspirada (ABC).
Usa unicamente datos_evaluacion/val.csv para la funcion objetivo.
3 ejecuciones independientes con semillas distintas.

Cada evaluacion de f(x) traduce todo el conjunto de validacion por lotes (no oracion por oracion) usando el componente NMT puro -- el reranker
completo con EBMT se reserva para el experimento 3 (demasiado costoso evaluarlo miles de veces dentro de ABC, 
luego mi laptop explota y no tengo paciencia suficiente).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

from src.rutas import ruta, MODELO_AJUSTADO
_ruta_ajustado = MODELO_AJUSTADO
from src.modelo.traductor import TraductorEsRu, ConfigDecodificacion
from src.abc.abc_optimizer import ColoniaABC
from src.abc.funcion_objetivo import f_objetivo, fitness_desde_f

with open(ruta("configs", "abc_search_space.json")) as f:
    ESPACIO = json.load(f)

NOMBRES_PARAMETROS = list(ESPACIO["parametros"].keys())
LIMITES = [(v["min"], v["max"]) for v in ESPACIO["parametros"].values()]

val = pd.read_csv(ruta("datos_evaluacion", "val.csv")).sample(n=300, random_state=42).reset_index(drop=True)

_ruta_modelo = _ruta_ajustado if os.path.isdir(_ruta_ajustado) else "Helsinki-NLP/opus-mt-es-ru"
print(f"Usando modelo: {_ruta_modelo}")
traductor = TraductorEsRu(ruta_modelo=_ruta_modelo)

_contador_evaluaciones = {"n": 0}


def vector_a_config(x: list[float]) -> ConfigDecodificacion:
    d = dict(zip(NOMBRES_PARAMETROS, x))
    return ConfigDecodificacion(
        num_beams=max(1, round(d["num_beams"])),
        length_penalty=d["length_penalty"],
        repetition_penalty=d["repetition_penalty"],
        max_length=128,
    )


def evaluar(x: list[float]) -> float:
    config = vector_a_config(x)
    traducciones = traductor.traducir_lote(val["es"].tolist(), config)
    _contador_evaluaciones["n"] += 1
    if _contador_evaluaciones["n"] % 5 == 0:
        print(f"  NFE={_contador_evaluaciones['n']}/330")
    return f_objetivo(traducciones, val["ru"].tolist())


def registrar_fabrica(ruta_csv: str, semilla: int):
    archivo = open(ruta_csv, "w", newline="", encoding="utf-8")
    escritor = csv.writer(archivo)
    escritor.writerow(["ejecucion", "semilla", "iteracion", "fuente_id", "parametros",
                        "objetivo", "mejor_acumulado", "tiempo"])

    def callback(id_ejecucion, iteracion, fuentes, mejor_x, mejor_f):
        t = time.time()
        for idx, fuente in enumerate(fuentes):
            escritor.writerow([id_ejecucion, semilla, iteracion, idx,
                                json.dumps(fuente.x), fuente.f, mejor_f, t])
        archivo.flush()
        print(f"  [ejecucion {id_ejecucion} | semilla {semilla}] iter {iteracion}: mejor_f={mejor_f:.4f}")

    return callback, archivo


def main():
    semillas = ESPACIO["presupuesto"]["semillas"]
    curvas = []
    mejor_global_x, mejor_global_f = None, float("inf")

    for i, semilla in enumerate(semillas, start=1):
        print(f"\n=== Ejecucion {i}/{len(semillas)} (semilla {semilla}) ===")
        ruta_csv = ruta("experimentos", "registros", f"experimento2_run{i}.csv")
        callback, archivo = registrar_fabrica(ruta_csv, semilla)
        colonia = ColoniaABC(
            limites=LIMITES,
            funcion_evaluar=evaluar,
            funcion_fitness=fitness_desde_f,
            SN=ESPACIO["presupuesto"]["fuentes_SN"],
            max_iteraciones=ESPACIO["presupuesto"]["iteraciones"],
            semilla=semilla,
            registrar_callback=callback,
        )
        mejor_x, mejor_f = colonia.ejecutar(id_ejecucion=i)
        archivo.close()

        if mejor_f < mejor_global_f:
            mejor_global_f, mejor_global_x = mejor_f, mejor_x

        df_run = pd.read_csv(ruta_csv)
        mejor_por_iter = df_run.groupby("iteracion")["mejor_acumulado"].min()
        curvas.append((f"Ejecución {i} (semilla {semilla})", mejor_por_iter))

    plt.figure()
    for etiqueta, serie in curvas:
        plt.plot(serie.index, serie.values, label=etiqueta)
    plt.xlabel("Iteración")
    plt.ylabel("Mejor valor acumulado de f(x) = 100 - chrF")
    plt.title("Convergencia ABC (3 ejecuciones)")
    plt.legend()
    plt.savefig(ruta("experimentos", "graficas", "convergencia.png"))

    mejor_dict = dict(zip(NOMBRES_PARAMETROS, mejor_global_x))
    with open(ruta("experimentos", "configuraciones", "config_optimizada_abc.json"), "w", encoding="utf-8") as f:
        json.dump(mejor_dict, f, ensure_ascii=False, indent=2)

    print("\nMejor configuración encontrada:", mejor_dict, "| f =", mejor_global_f,
        f"(chrF = {100 - mejor_global_f:.2f})")


if __name__ == "__main__":
    main()
