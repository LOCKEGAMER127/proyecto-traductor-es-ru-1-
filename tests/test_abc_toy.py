"""
Prueba del algoritmo ABC (src/abc/abc_optimizer.py) sobre una funcion sintetica (esfera desplazada), SIN necesitar el modelo de traduccion ni
conexion a internet.

Proposito: Reconocer si hay que realizr ajustes a ABC o al modelo.
Si el modelo converge aqui el pedillo esta en el modelo
Si no convergue aqui el pedillo esta en ABC
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.abc.abc_optimizer import ColoniaABC

# Funcion esfera desplazada: minimo global conocido en x* = (3, -2, 1)
OPTIMO = [3.0, -2.0, 1.0]


def funcion_esfera(x):
    return sum((xi - oi) ** 2 for xi, oi in zip(x, OPTIMO))


def fitness(f):
    return 1.0 / (1.0 + f) if f >= 0 else 1.0 + abs(f)


def main():
    limites = [(-10, 10), (-10, 10), (-10, 10)]

    colonia = ColoniaABC(
        limites=limites,
        funcion_evaluar=funcion_esfera,
        funcion_fitness=fitness,
        SN=20,
        max_iteraciones=40,
        semilla=42,
    )
    mejor_x, mejor_f = colonia.ejecutar(id_ejecucion=1)

    print(f"Optimo real:      {OPTIMO}")
    print(f"Mejor encontrado: {[round(v, 4) for v in mejor_x]}")
    print(f"f(mejor) = {mejor_f:.6f}  (deberia acercarse a 0)")

    error = funcion_esfera(mejor_x)
    assert error < 0.5, f"ABC no convergio lo suficiente (error={error:.4f}). Revisar abc_optimizer.py."
    print("\n✅ El algoritmo ABC converge correctamente sobre un problema conocido.")


if __name__ == "__main__":
    main()
