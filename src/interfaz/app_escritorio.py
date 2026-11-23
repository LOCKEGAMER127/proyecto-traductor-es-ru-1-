"""
Interfaz de escritorio interactiva (CustomTkinter) para el traductor es-ru.

Requisitos que debe cumplir:
- Aceptar texto en espanol.
- Ejecutar el modelo (con la config seleccionada: base u optimizada por ABC).
- Mostrar idioma de destino y traduccion.
- Identificar la configuracion usada.
- Funcionar con textos nuevos.
"""

import json
import customtkinter as ctk
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.rutas import ruta
from src.rutas import ruta
from src.combinacion.pipeline import TraductorHibrido
from src.combinacion.reranker import PesosCombinacion
from src.modelo.traductor import ConfigDecodificacion

RUTA_MODELO_AJUSTADO = r"C:/modelos/opus-mt-es-ru-ajustado"
RUTA_MODELO_BASE = "Helsinki-NLP/opus-mt-es-ru"
RUTA_CONFIG_BASE = ruta("configs", "decoding_base.json")
RUTA_CONFIG_OPTIMA = ruta("experimentos", "configuraciones", "config_optimizada_abc.json")
RUTA_TRAIN_CSV = ruta("datos_evaluacion", "train.csv")


def _modelo_disponible(ruta_carpeta: str) -> bool:
    import os
    return os.path.isdir(ruta_carpeta) and os.path.exists(os.path.join(ruta_carpeta, "config.json"))


def cargar_config(ruta_json: str, pesos_por_defecto=True) -> tuple[ConfigDecodificacion, PesosCombinacion]:
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)
    config = ConfigDecodificacion(
        num_beams=round(datos.get("num_beams", 4)),
        length_penalty=datos.get("length_penalty", 1.0),
        repetition_penalty=datos.get("repetition_penalty", 1.0),
        max_length=int(datos.get("max_length", 128)),
    )
    pesos = PesosCombinacion(
        w_nmt=datos.get("w_nmt", 1.0),
        w_ebmt=datos.get("w_ebmt", 0.0),
        w_reglas=datos.get("w_reglas", 0.0),
    )
    return config, pesos


class AppTraductor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Traductor Español → Ruso (ABC)")
        self.geometry("680x520")

        ruta_modelo = RUTA_MODELO_AJUSTADO if _modelo_disponible(RUTA_MODELO_AJUSTADO) else RUTA_MODELO_BASE
        import os
        self.traductor = TraductorHibrido(
            ruta_modelo=ruta_modelo,
            ruta_train_csv=RUTA_TRAIN_CSV if os.path.exists(RUTA_TRAIN_CSV) else None,
        )

        self.config_base, self.pesos_base = cargar_config(RUTA_CONFIG_BASE)
        try:
            self.config_optima, self.pesos_optima = cargar_config(RUTA_CONFIG_OPTIMA)
        except FileNotFoundError:
            self.config_optima, self.pesos_optima = None, None

        self._construir_widgets(ruta_modelo)

    def _construir_widgets(self, ruta_modelo_usada: str):
        ctk.CTkLabel(self, text=f"Modelo cargado: {ruta_modelo_usada}", font=("", 11)).pack(pady=(10, 0))

        ctk.CTkLabel(self, text="Texto en español:").pack(pady=(10, 0))
        self.entrada = ctk.CTkTextbox(self, height=100)
        self.entrada.pack(fill="x", padx=20)

        opciones = ["Configuración base"]
        if self.config_optima is not None:
            opciones.append("Configuración optimizada (ABC)")
        self.modo = ctk.CTkOptionMenu(self, values=opciones)
        self.modo.pack(pady=10)

        ctk.CTkButton(self, text="Traducir", command=self.traducir).pack(pady=5)

        ctk.CTkLabel(self, text="Idioma de destino: Ruso").pack()
        self.salida = ctk.CTkTextbox(self, height=100)
        self.salida.pack(fill="x", padx=20, pady=(0, 5))

        self.etiqueta_config = ctk.CTkLabel(self, text="Configuración usada: —")
        self.etiqueta_config.pack()
        self.etiqueta_origen = ctk.CTkLabel(self, text="Candidato ganador: —")
        self.etiqueta_origen.pack()

    def traducir(self):
        texto = self.entrada.get("1.0", "end").strip()
        if not texto:
            return

        if self.modo.get() == "Configuración optimizada (ABC)" and self.config_optima:
            config, pesos = self.config_optima, self.pesos_optima
            etiqueta = "Optimizada por ABC"
        else:
            config, pesos = self.config_base, self.pesos_base
            etiqueta = "Base (configs/decoding_base.json)"

        resultado = self.traductor.traducir(texto, config, pesos)

        self.salida.delete("1.0", "end")
        self.salida.insert("1.0", resultado["traduccion"])
        self.etiqueta_config.configure(text=f"Configuración usada: {etiqueta}")
        self.etiqueta_origen.configure(text=f"Candidato ganador: {resultado['origen_ganador']}")


if __name__ == "__main__":
    app = AppTraductor()
    app.mainloop()
