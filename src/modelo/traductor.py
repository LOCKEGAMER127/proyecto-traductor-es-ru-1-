"""
Envoltura robusta del modelo NMT (Helsinki-NLP/opus-mt-es-ru o el ajustado).
"""

import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _detectar_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ConfigDecodificacion:
    # Vector de parametros que ABC optimiza. Debe coincidir con las llaves de configs/abc_search_space.json.
    num_beams: int = 4
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    max_length: int = 128

    def to_dict(self) -> dict:
        return {
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_length": self.max_length,
        }


class TraductorEsRu:
    # Carga el modelo una sola vez (perezoso) y traduce por lotes reales, preservando siempre el orden de entrada.

    def __init__(self, ruta_modelo: str = "Helsinki-NLP/opus-mt-es-ru", device: str = None, tamano_lote: int = 16):
        self.ruta_modelo = ruta_modelo
        self.device = device or _detectar_device()
        self.tamano_lote = tamano_lote
        self.tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
        self.modelo = AutoModelForSeq2SeqLM.from_pretrained(ruta_modelo).to(self.device)
        self.modelo.eval()

    @torch.no_grad()
    def _generar_lote(self, textos: list[str], config: ConfigDecodificacion, n_best: int = 1, devolver_scores: bool = False):
        entradas = self.tokenizer(
            textos, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=True,
        ).to(self.device)

        salida = self.modelo.generate(
            **entradas,
            num_beams=max(config.num_beams, n_best),
            num_return_sequences=n_best,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            max_length=config.max_length,
            early_stopping=True,
            output_scores=devolver_scores,
            return_dict_in_generate=devolver_scores,
        )

        if devolver_scores:
            secuencias = salida.sequences
            # sequences_scores: log-prob normalizada por longitud (una por hipotesis)
            puntajes = salida.sequences_scores.tolist()
        else:
            secuencias = salida
            puntajes = [0.0] * (len(textos) * n_best)

        textos_decodificados = self.tokenizer.batch_decode(secuencias, skip_special_tokens=True)
        return textos_decodificados, puntajes

    def traducir(self, texto: str, config: ConfigDecodificacion, n_best: int = 1):
        # Traduce una sola oracion. Para evaluar conjuntos completos usar traducir_lote (mucho mas rapido y menos propenso a errores).
        textos, puntajes = self._generar_lote([texto], config, n_best=n_best, devolver_scores=(n_best > 1))
        if n_best == 1:
            return textos[0]
        return list(zip(textos, puntajes))

    def traducir_lote(self, textos: list[str], config: ConfigDecodificacion) -> list[str]:
        """
        Traduce una lista completa respetando el orden, en mini-lotes de tamano self.tamano_lote. Esto es lo que deben usar los experimentos
        1, 2 y 3 (nunca un loop oracion por oracion sobre miles de filas).
        """
        resultados = []
        for i in range(0, len(textos), self.tamano_lote):
            mini_lote = textos[i:i + self.tamano_lote]
            traducidos, _ = self._generar_lote(mini_lote, config, n_best=1, devolver_scores=False)
            resultados.extend(traducidos)
        assert len(resultados) == len(textos), "Se perdio el orden o se omitieron oraciones en el lote."
        return resultados

    def traducir_n_best(self, texto: str, config: ConfigDecodificacion, n_best: int = 3):
        # Devuelve [(texto_candidato, logprob_normalizada), ...] para que el reranker (src/combinacion/reranker.py) combine con EBMT y reglas.
        return self.traducir(texto, config, n_best=n_best)
