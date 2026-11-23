"""
Construye el corpus es-ru para el proyecto de traducción (TLN + Algoritmos Bioinspirados).
Fuente principal: Helsinki-NLP/un_pc, config "es-ru" (Corpus Paralelo de la ONU).
Uso:
    python build_corpus.py --n_samples 150000 --seed 42 --min_len 3 --max_len 120

Salidas (en ./datos_evaluacion/):
    train.csv, val.csv, test.csv   -> columnas: es, ru
    corpus_log.json                -> conteos antes/después, fuente, licencia
    ejemplos_muestra.csv           -> 5 pares de ejemplo (evidencia pedida)
"""

import argparse
import json
import os
import random
import hashlib
from datasets import load_dataset
import pandas as pd


def normalizar(texto: str) -> str:
    # Normalización básica: espacios, comillas raras, NBSP.
    if texto is None:
        return ""
    texto = texto.replace("\xa0", " ")
    texto = " ".join(texto.split())
    return texto.strip()


def hash_par(es: str, ru: str) -> str:
    # Hash para detectar duplicados y variantes casi-idénticas.
    clave = (es.lower().strip() + "|||" + ru.lower().strip())
    return hashlib.md5(clave.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=150_000, help="Cuántos pares tomar del stream de un_pc (antes de limpiar).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_len", type=int, default=3, help="Mínimo de palabras por oración.")
    ap.add_argument("--max_len", type=int, default=120, help="Máximo de palabras por oración.")
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--val_size", type=int, default=1000)
    ap.add_argument("--out_dir", type=str, default="datos_evaluacion")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    log = {
        "fuente": "Helsinki-NLP/un_pc",
        "config": "es-ru",
        "licencia": "liberal / open (ver dataset card en HuggingFace)",
        "direccion": "https://huggingface.co/datasets/Helsinki-NLP/un_pc",
        "cantidad_original_tomada": 0,
        "cantidad_tras_dedup": 0,
        "cantidad_tras_filtro_longitud": 0,
        "cantidad_final_train": 0,
        "cantidad_final_val": 0,
        "cantidad_final_test": 0,
    }

    print(f"Descargando {args.n_samples} pares de Helsinki-NLP/un_pc (es-ru) en streaming...")
    ds = load_dataset("Helsinki-NLP/un_pc", "es-ru", split="train", streaming=True)

    pares = []
    for i, ejemplo in enumerate(ds.take(args.n_samples)):
        t = ejemplo["translation"]
        es = normalizar(t.get("es", ""))
        ru = normalizar(t.get("ru", ""))
        if es and ru:
            pares.append((es, ru))
    log["cantidad_original_tomada"] = len(pares)
    print(f"  -> {len(pares)} pares descargados.")

    # --- Deduplicación ---
    vistos = set()
    unicos = []
    for es, ru in pares:
        h = hash_par(es, ru)
        if h not in vistos:
            vistos.add(h)
            unicos.append((es, ru))
    log["cantidad_tras_dedup"] = len(unicos)
    print(f"  -> {len(unicos)} pares tras eliminar duplicados.")

    # --- Filtro por longitud (palabras) ---
    filtrados = [
        (es, ru) for es, ru in unicos
        if args.min_len <= len(es.split()) <= args.max_len
        and args.min_len <= len(ru.split()) <= args.max_len
    ]
    log["cantidad_tras_filtro_longitud"] = len(filtrados)
    print(f"  -> {len(filtrados)} pares tras filtrar por longitud.")

    # --- Split 80/10/10 SIN fuga: se baraja una sola vez y se corta ---
    random.shuffle(filtrados)
    n_test = min(args.test_size, max(100, int(0.1 * len(filtrados))))
    n_val = min(args.val_size, max(100, int(0.1 * len(filtrados))))

    test = filtrados[:n_test]
    val = filtrados[n_test:n_test + n_val]
    train = filtrados[n_test + n_val:]

    log["cantidad_final_train"] = len(train)
    log["cantidad_final_val"] = len(val)
    log["cantidad_final_test"] = len(test)

    def a_csv(nombre, datos):
        df = pd.DataFrame(datos, columns=["es", "ru"])
        ruta = os.path.join(args.out_dir, nombre)
        df.to_csv(ruta, index=False, encoding="utf-8")
        print(f"  Guardado {ruta} ({len(df)} filas)")

    a_csv("train.csv", train)
    a_csv("val.csv", val)
    a_csv("test.csv", test)

    # --- Evidencia: 5 ejemplos ---
    muestra = random.sample(train, min(5, len(train)))
    pd.DataFrame(muestra, columns=["es", "ru"]).to_csv(
        os.path.join(args.out_dir, "ejemplos_muestra.csv"), index=False, encoding="utf-8"
    )

    with open(os.path.join(args.out_dir, "corpus_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("\nResumen final:")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
