"""
Chequeo rapido de cordura ANTES de correr los 5 experimentos completos.
Traduce 3 oraciones conocidas y calcula chrF contra referencias de Tatoeba (dominio facil, frases cortas). Si aqui el chrF sale muy bajo.
Hay que revisar el cargado del modelo 
Vuelvo a pasar por aqui y digo: que bueno que no hizo falta 
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modelo.traductor import TraductorEsRu, ConfigDecodificacion
from src.metricas.calcular_metricas import calcular_bleu_chrf

# Pares de prueba conocidos (frases simples, cualquier modelo razonable deberia traducirlas bien)
CASOS = [
    ("Hola, ¿cómo estás?", "Привет, как дела?"),
    ("El gato está en la casa.", "Кошка в доме."),
    ("Buenos días, gracias por venir.", "Доброе утро, спасибо что пришли."),
]


def main():
    print("Cargando Helsinki-NLP/opus-mt-es-ru (sin ajustar) para diagnostico...")
    traductor = TraductorEsRu(ruta_modelo="Helsinki-NLP/opus-mt-es-ru")
    config = ConfigDecodificacion(num_beams=4, length_penalty=1.0, repetition_penalty=1.0, max_length=64)

    entradas = [es for es, _ in CASOS]
    referencias = [ru for _, ru in CASOS]

    traducciones = traductor.traducir_lote(entradas, config)

    print("\nResultados:")
    for es, ref, trad in zip(entradas, referencias, traducciones):
        print(f"  ES:  {es}")
        print(f"  REF: {ref}")
        print(f"  OUT: {trad}")
        print()

    metricas = calcular_bleu_chrf(traducciones, referencias)
    print(f"chrF sobre estos 3 casos simples: {metricas['chrf']:.2f} / 100")

    if metricas["chrf"] < 30:
        print("\n chrF sospechosamente bajo incluso en frases triviales. Algo salio mal, no puede ser :( ")
    else:
        print("\n El modelo base responde razonablemente. Si jalo tú")


if __name__ == "__main__":
    main()
