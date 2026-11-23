"""
Componente de ejemplo (EBMT): memoria de traduccion.

Dos modos:
- 'tfidf'      -> rapido, sin dependencias pesadas, buena linea base.
- 'embeddings' -> usa sentence-transformers multilingue (mejor para parafrasis/sinonimos), requiere el paquete instalado.

Si una oracion de entrada es muy parecida a una ya vista en entrenamiento, su traduccion conocida se ofrece como candidata adicional al reranker.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MemoriaDeTraduccion:
    def __init__(self, ruta_train_csv: str, umbral_similitud: float = 0.6, modo: str = "tfidf"):
        self.umbral = umbral_similitud
        self.modo = modo
        self.df = pd.read_csv(ruta_train_csv).dropna(subset=["es", "ru"]).reset_index(drop=True)

        if modo == "tfidf":
            self.vectorizador = TfidfVectorizer()
            self.matriz = self.vectorizador.fit_transform(self.df["es"].astype(str))
        elif modo == "embeddings":
            from sentence_transformers import SentenceTransformer
            self.modelo_emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.matriz = self.modelo_emb.encode(
                self.df["es"].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=False
            )
        else:
            raise ValueError(f"Modo no soportado: {modo}")

    def buscar_mas_similar(self, texto: str):
        # Devuelve (traduccion_recuperada, similitud) o (None, 0.0) si no supera el umbral.
        if self.modo == "tfidf":
            vector = self.vectorizador.transform([texto])
            similitudes = cosine_similarity(vector, self.matriz)[0]
        else:
            vector = self.modelo_emb.encode([texto], normalize_embeddings=True)
            similitudes = (self.matriz @ vector[0])

        idx = int(np.argmax(similitudes))
        mejor_sim = float(similitudes[idx])
        if mejor_sim >= self.umbral:
            return self.df.iloc[idx]["ru"], mejor_sim
        return None, 0.0
