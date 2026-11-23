"""
Implementacion del algoritmo de Colonia Artificial de Abejas (ABC),
siguiendo la Guia Tecnica de Implementacion - Algoritmo ABC (Karaboga, 2005;
Karaboga & Basturk, 2007).

Fases: A) Inicializacion, B) Empleadas, C) Observadoras, D) Exploradoras.

Este optimizador es como terraplanista en 2026, no te cree nada: recibe una funcion `evaluar(x)`
que traduce el conjunto de validacion con los parametros x, calcula chrF y regresa f(x). Asi se conecta directamente con el traductor real.
"""

import random
from dataclasses import dataclass, field


@dataclass
class Fuente:
    x: list[float]
    f: float = None
    fit: float = None
    trial: int = 0


class ColoniaABC:
    def __init__(
        self,
        limites: list[tuple[float, float]],   # [(L_1,U_1), ..., (L_D,U_D)]
        funcion_evaluar,                       # x -> f(x)   (minimizar)
        funcion_fitness,                       # f(x) -> fit > 0
        SN: int = 20,
        limite_abandono: int = None,
        max_iteraciones: int = 10,
        semilla: int = 42,
        registrar_callback=None,               # callback(ejecucion, iteracion, fuentes) -> None
    ):
        self.limites = limites
        self.D = len(limites)
        self.evaluar = funcion_evaluar
        self.fitness = funcion_fitness
        self.SN = SN
        self.limite_abandono = limite_abandono or int(self.SN * self.D)
        self.max_iteraciones = max_iteraciones
        self.rng = random.Random(semilla)
        self.registrar_callback = registrar_callback

        self.fuentes: list[Fuente] = []
        self.mejor_x = None
        self.mejor_f = float("inf")

    # Inicializacion
    def _generar_aleatoria(self) -> list[float]:
        return [
            self.rng.uniform(low, high) for (low, high) in self.limites
        ]

    def inicializar(self):
        self.fuentes = []
        for _ in range(self.SN):
            x = self._generar_aleatoria()
            f = self.evaluar(x)
            fit = self.fitness(f)
            self.fuentes.append(Fuente(x=x, f=f, fit=fit, trial=0))
        self._actualizar_mejor_global()

    # Operador de vecindad (comun a empleadas/observadoras) 
    def _generar_vecina(self, i: int) -> list[float]:
        k = i
        while k == i:
            k = self.rng.randrange(self.SN)
        j = self.rng.randrange(self.D)
        phi = self.rng.uniform(-1, 1)

        nuevo_x = list(self.fuentes[i].x)
        valor = self.fuentes[i].x[j] + phi * (self.fuentes[i].x[j] - self.fuentes[k].x[j])
        low, high = self.limites[j]
        nuevo_x[j] = min(max(valor, low), high)  # clamping
        return nuevo_x

    def _seleccion_voraz(self, i: int, x_candidato: list[float]):
        f_candidato = self.evaluar(x_candidato)
        fit_candidato = self.fitness(f_candidato)
        if fit_candidato > self.fuentes[i].fit:
            self.fuentes[i] = Fuente(x=x_candidato, f=f_candidato, fit=fit_candidato, trial=0)
        else:
            self.fuentes[i].trial += 1

    # Fase B: Empleadas
    def _fase_empleadas(self):
        for i in range(self.SN):
            candidata = self._generar_vecina(i)
            self._seleccion_voraz(i, candidata)

    # Fase C: Observadoras
    def _probabilidades(self) -> list[float]:
        suma_fit = sum(f.fit for f in self.fuentes)
        return [f.fit / suma_fit for f in self.fuentes]

    def _fase_observadoras(self):
        probabilidades = self._probabilidades()
        for _ in range(self.SN):
            i = self.rng.choices(range(self.SN), weights=probabilidades, k=1)[0]
            candidata = self._generar_vecina(i)
            self._seleccion_voraz(i, candidata)

    # Fase D: Exploradoras
    def _fase_exploradoras(self):
        idx_max = max(range(self.SN), key=lambda i: self.fuentes[i].trial)
        if self.fuentes[idx_max].trial >= self.limite_abandono:
            nuevo_x = self._generar_aleatoria()
            f = self.evaluar(nuevo_x)
            fit = self.fitness(f)
            self.fuentes[idx_max] = Fuente(x=nuevo_x, f=f, fit=fit, trial=0)

    def _actualizar_mejor_global(self):
        for f in self.fuentes:
            if f.f < self.mejor_f:
                self.mejor_f = f.f
                self.mejor_x = list(f.x)

    # Ciclo principal
    def ejecutar(self, id_ejecucion: int = 1):
        self.inicializar()
        for iteracion in range(1, self.max_iteraciones + 1):
            self._fase_empleadas()
            self._fase_observadoras()
            self._fase_exploradoras()
            self._actualizar_mejor_global()

            if self.registrar_callback:
                self.registrar_callback(id_ejecucion, iteracion, self.fuentes, self.mejor_x, self.mejor_f)

        return self.mejor_x, self.mejor_f
