"""
TraductorHibrido: amarra reglas + modelo (NMT) + EBMT + reranker en un solo objeto. 
Esta es la pieza que deben usar tanto la GUI como los experimentos 1-5 y la funcion objetivo de ABC, para que todos midan exactamente el mismo sistema.
"""

from src.modelo.traductor import TraductorEsRu, ConfigDecodificacion
from src.ejemplo.memoria_traduccion import MemoriaDeTraduccion
from src.combinacion.reranker import PesosCombinacion, puntuar_candidatos, elegir_mejor_candidato
from src.reglas.reglas_es_ru import (
    cargar_diccionario_forzado,
    pipeline_reglas_pre,
    pipeline_reglas_post,
)


class TraductorHibrido:
    def __init__(
        self,
        ruta_modelo: str = "Helsinki-NLP/opus-mt-es-ru",
        ruta_train_csv: str = None,
        modo_ebmt: str = "tfidf",
        tamano_lote: int = 16,
    ):
        self.traductor = TraductorEsRu(ruta_modelo=ruta_modelo, tamano_lote=tamano_lote)
        self.memoria = (
            MemoriaDeTraduccion(ruta_train_csv, modo=modo_ebmt) if ruta_train_csv else None
        )
        self.diccionario = cargar_diccionario_forzado()

    def traducir(
        self,
        texto: str,
        config: ConfigDecodificacion,
        pesos: PesosCombinacion = PesosCombinacion(),
        n_best: int = 3,
    ) -> dict:
        """Traduce una sola oracion combinando los tres componentes.
        Devuelve un dict con el texto final y metadatos (util para la GUI, que debe 'identificar la configuracion usada')."""
        texto_normalizado = pipeline_reglas_pre(texto)

        candidatos_nmt = self.traductor.traducir_n_best(texto_normalizado, config, n_best=n_best)

        texto_ebmt, similitud_ebmt = (None, 0.0)
        if self.memoria is not None:
            texto_ebmt, similitud_ebmt = self.memoria.buscar_mas_similar(texto_normalizado)

        candidatos = puntuar_candidatos(
            candidatos_nmt, texto_ebmt, similitud_ebmt, self.diccionario, pesos
        )
        mejor = elegir_mejor_candidato(candidatos)

        texto_final = pipeline_reglas_post(mejor["texto"], self.diccionario)

        return {
            "traduccion": texto_final,
            "origen_ganador": mejor["origen"],
            "candidatos_evaluados": candidatos,
            "config_decodificacion": config.to_dict(),
            "pesos_combinacion": vars(pesos),
        }

    def traducir_lote_simple(self, textos: list[str], config: ConfigDecodificacion) -> list[str]:
        """Traduccion por lotes SOLO con el componente NMT (sin EBMT ni reranker) -- usar dentro de la funcion objetivo de ABC, donde se
        evaluan cientos/miles de oraciones de validacion por iteracion y la velocidad importa mas que exprimir cada decima de chrF.
        El reranker completo (self.traducir) se reserva para el experimento final de comparacion y para la GUI."""
        textos_normalizados = [pipeline_reglas_pre(t) for t in textos]
        return self.traductor.traducir_lote(textos_normalizados, config)
