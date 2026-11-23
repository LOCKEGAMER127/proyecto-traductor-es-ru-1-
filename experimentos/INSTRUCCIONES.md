# Instrucciones para reproducir los experimentos

## Orden de ejecucion (desde la raiz del repositorio)

1. `python src/corpus/build_corpus.py --n_samples 130000 --test_size 1000 --val_size 1000`
   Genera `datos_evaluacion/train.csv` (128,000), `val.csv` (1,000) y `test.csv` (1,000).

2. `python src/modelo/finetune.py`
   Ajusta Helsinki-NLP/opus-mt-es-ru sobre train.csv. Guarda el modelo en
   `modelos_guardados/opus-mt-es-ru-ajustado/`.

3. `cd experimentos && python experimento1_modelo_base.py`
   Congela y evalua el modelo base. Salida: `tablas/experimento1.json`.

4. `python experimento2_optimizacion_abc.py`
   Corre ABC (3 semillas, 10 iteraciones, 20 fuentes) usando SOLO val.csv.
   Salida: `registros/experimento2_run{1,2,3}.csv`, `graficas/convergencia.png`,
   `configuraciones/config_optimizada_abc.json`.

5. `python experimento3_comparacion.py`
   Compara base vs. optimizada sobre test.csv (una sola vez).

6. `python experimento4_comparacion_cualitativa.py`
7. `python experimento5_traductor_externo.py`

## Version de bibliotecas
Ver `requirements.txt` en la raiz del repositorio (congelar versiones exactas
con `pip freeze > requirements_congelado.txt` antes de la entrega final).

## Ruta de los datos
Los conjuntos de datos se generan localmente a partir de
`Helsinki-NLP/un_pc` (config es-ru) via la libreria `datasets`. No se
redistribuyen directamente en el repositorio por tamaño; `build_corpus.py`
los reconstruye de forma reproducible con semilla fija (--seed 42).
