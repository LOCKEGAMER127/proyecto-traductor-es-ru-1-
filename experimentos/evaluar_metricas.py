"""
Punto de entrada unico para regenerar TODAS las tablas y graficas del articulo a partir de los registros ya guardados (no vuelve a traducir).
Util para la revision/reproducibilidad.
"""
import json
import glob


def main():
    print("Tablas disponibles en tablas/:")
    for ruta in sorted(glob.glob("tablas/*")):
        print(" -", ruta)

    print("\nRegistros disponibles en registros/:")
    for ruta in sorted(glob.glob("registros/*")):
        print(" -", ruta)

    print("\nGraficas disponibles en graficas/:")
    for ruta in sorted(glob.glob("graficas/*")):
        print(" -", ruta)

    # si se requiere recalcular (no solo listar), leer cada CSV/JSON y volver a correr src/metricas/calcular_metricas.py sobre ellos.


if __name__ == "__main__":
    main()
