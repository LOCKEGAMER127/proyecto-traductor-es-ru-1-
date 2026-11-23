# Traductor Español → Ruso con optimización ABC

Proyecto integrador de **Tecnologías del Lenguaje Natural** y **Algoritmos Bioinspirados**.

## Idioma de destino
Ruso (es → ru)

## Algoritmo bioinspirado asignado
Colonia Artificial de Abejas (ABC)

## Arquitectura (resumen)
1. **Reglas** (`src/reglas/`): normalización, diccionario forzado de términos.
2. **Modelo ajustado** (`src/modelo/`): fine-tuning de `Helsinki-NLP/opus-mt-es-ru`.
3. **Ejemplo / memoria de traducción** (`src/ejemplo/`): recuperación por similitud sobre el corpus de entrenamiento.
4. **Combinación** (`src/combinacion/`): reponderación log-lineal del n-best generado por el modelo.
5. **ABC** (`src/abc/`): optimiza parámetros reales de decodificación (`num_beams`,
   `length_penalty`, `repetition_penalty`) y los pesos de la combinación, maximizando chrF
   sobre el conjunto de validación.
6. **Interfaz** (`src/interfaz/`): aplicación de escritorio interactiva (Tkinter/CustomTkinter).

Ver detalle de cada módulo en los docstrings de cada archivo.

## Requisitos
Ver `requirements.txt`. Python 3.10+.

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Comandos de ejecución

### 1. Construir el corpus
```bash
python src/corpus/build_corpus.py --n_samples 130000 --test_size 1000 --val_size 1000
```
Genera `datos_evaluacion/train.csv` (~128k), `val.csv` (1000) y `test.csv` (1000).

### 2. Verificar que el modelo funciona (antes de correr todo)
```bash
python scripts/diagnostico.py
```
Traduce 3 frases simples y reporta chrF. Si aquí el resultado ya es malo,
el problema está en la carga del modelo, no en tu corpus ni en ABC.

### 2.1 Probar que el algoritmo ABC converge (sin modelo, sin internet)
```bash
python -m tests.test_abc_toy
```

### 3. Ajustar el modelo base
```bash
python src/modelo/finetune.py
```

### 3. Ejecutar la interfaz de escritorio
```bash
python src/interfaz/app_escritorio.py
```

Durante la ejecucion de este ultimo puede sirguir un error debido al nombre de la ruta donde el proyecto existe.
Para solucionar esto copie el archivo donde se guardo el modelo ajustado a una carpeto que solo contenga caracteres del sistema ASCII

```bash
python src/interfaz/app_escritorio.py                                    
xcopy /E /I "modelos_guardados\opus-mt-es-ru-ajustado" C:\modelos\opus-mt-es-ru-ajustado # <Ubicacion actual (no modificar)> <Nueva ubicacion (Usar solo caracteres del sistema ASCII)>
```

### 4. Reproducir los experimentos
```bash
cd experimentos
python experimento1_modelo_base.py
python experimento2_optimizacion_abc.py
python experimento3_comparacion.py
python experimento4_comparacion_cualitativa.py
python experimento5_traductor_externo.py
```
Ver `experimentos/INSTRUCCIONES.md` para el orden y los parámetros de cada paso.

## Corpus utilizado
- **Fuente:** Helsinki-NLP/un_pc (config es-ru), United Nations Parallel Corpus (1990-2014).
- **Licencia:** liberal / abierta (ver dataset card en HuggingFace).
- **Cantidad usada:** 128,000 entrenamiento / 1,000 validación / 1,000 prueba (muestreo
  aleatorio con semilla fija, deduplicado, sin fuga entre conjuntos).

## Modelo base
- **Nombre:** Helsinki-NLP/opus-mt-es-ru (Marian / Transformer-align), ajustado sobre el
  corpus propio.

## Artículo
Enlace de consulta (Overleaf, View Link): *(agregar aquí)*

## Licencia y datos
No se publican contraseñas ni credenciales. Ver `data/` para rutas de descarga de corpus
y modelos cuando no puedan incluirse directamente por tamaño o licencia.
