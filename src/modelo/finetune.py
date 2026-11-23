"""
Ajuste (fine-tuning) de Helsinki-NLP/opus-mt-es-ru sobre el corpus propio, con chrF calculado durante el entrenamiento (sobre validacion) para
seleccionar el mejor checkpoint, no solo minimizar loss.
"""

import argparse
import json
import numpy as np
import pandas as pd
import sacrebleu
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.rutas import ruta


def cargar_csv_como_dataset(ruta_csv: str) -> Dataset:
    df = pd.read_csv(ruta_csv).dropna(subset=["es", "ru"])
    df = df[df["es"].str.strip().astype(bool) & df["ru"].str.strip().astype(bool)]
    return Dataset.from_pandas(df[["es", "ru"]].reset_index(drop=True))


def construir_tokenizador_lote(tokenizer, max_length):
    def tokenizar_lote(lote):
        entradas = tokenizer(lote["es"], truncation=True, max_length=max_length)
        etiquetas = tokenizer(text_target=lote["ru"], truncation=True, max_length=max_length)
        entradas["labels"] = etiquetas["input_ids"]
        return entradas
    return tokenizar_lote


def construir_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        textos_pred = tokenizer.batch_decode(preds, skip_special_tokens=True)
        textos_ref = tokenizer.batch_decode(labels, skip_special_tokens=True)

        textos_pred = [t.strip() if t.strip() else " " for t in textos_pred]
        textos_ref = [t.strip() if t.strip() else " " for t in textos_ref]

        chrf = sacrebleu.corpus_chrf(textos_pred, [textos_ref]).score
        bleu = sacrebleu.corpus_bleu(textos_pred, [textos_ref]).score
        return {"chrf": chrf, "bleu": bleu}
    return compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo_base", default="Helsinki-NLP/opus-mt-es-ru")
    ap.add_argument("--train_csv", default=ruta("datos_evaluacion/train.csv"))
    ap.add_argument("--val_csv", default=ruta("datos_evaluacion/val.csv"))
    ap.add_argument("--salida", default=ruta("modelos_guardados/opus-mt-es-ru-ajustado"))
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--paciencia_early_stopping", type=int, default=2)
    ap.add_argument("--fp16", action="store_true", help="Usar precision mixta (requiere GPU NVIDIA).")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.modelo_base)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(args.modelo_base)

    tokenizar_lote = construir_tokenizador_lote(tokenizer, args.max_length)

    ds_train = cargar_csv_como_dataset(args.train_csv).map(tokenizar_lote, batched=True,
                                                            remove_columns=["es", "ru"])
    ds_val = cargar_csv_como_dataset(args.val_csv).map(tokenizar_lote, batched=True,
                                                        remove_columns=["es", "ru"])

    # Clave: padding de etiquetas -> -100 para que no afecte la perdida.
    collator = DataCollatorForSeq2Seq(tokenizer, model=modelo, label_pad_token_id=-100)

    args_entrenamiento = Seq2SeqTrainingArguments(
        output_dir=args.salida,
        num_train_epochs=args.epocas,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        fp16=args.fp16,
        logging_steps=50,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=modelo,
        args=args_entrenamiento,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=construir_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.paciencia_early_stopping)],
    )

    resultado_entrenamiento = trainer.train()
    metricas_finales = trainer.evaluate()

    trainer.save_model(args.salida)
    tokenizer.save_pretrained(args.salida)

    with open(f"{args.salida}/hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    with open(f"{args.salida}/metricas_finales.json", "w", encoding="utf-8") as f:
        json.dump(metricas_finales, f, ensure_ascii=False, indent=2)

    print("Entrenamiento terminado. Metricas finales de validacion:")
    print(json.dumps(metricas_finales, ensure_ascii=False, indent=2))
    print(f"Modelo ajustado guardado en {args.salida}")


if __name__ == "__main__":
    main()
