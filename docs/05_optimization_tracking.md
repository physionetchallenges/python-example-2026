# Seguimiento De Optimizaciones

Este documento registra las optimizaciones de tiempo de entrenamiento aplicadas sobre el código de ejemplo original del PhysioNet Challenge en `team_code.py`.

## Objetivo

Mantener un registro claro de los cambios respecto a la base proporcionada por la organización para que el equipo pueda:

- entender qué optimizaciones se probaron;
- medir su efecto sobre el flujo smoke;
- identificar qué cambios merece la pena conservar;
- revertir cambios concretos si la submission se comporta distinto en el entorno del Challenge.

## Línea Base

- Fuente de la línea base: implementación de ejemplo proporcionada por la organización en `team_code.py`.
- Tiempo observado de entrenamiento smoke con `./run.sh train-dev`: alrededor de 22 segundos.
- Comportamiento base: extracción secuencial de features, lecturas repetidas de CSV, recarga repetida de reglas de renombrado de canales y carga no utilizada de anotaciones humanas durante entrenamiento.

## Cambios Aplicados

### 1. Eliminación de la carga no utilizada de anotaciones humanas en entrenamiento

Cambio:

- Se eliminó la carga de `human_annotations` dentro de `train_model`.
- Se dejó intacta la función auxiliar `extract_human_annotations_features`.

Motivo:

- El vector final de entrenamiento solo concatenaba features demográficas, fisiológicas y algorítmicas.
- Las features de anotaciones humanas se calculaban, pero nunca se incluían en el `np.hstack(...)` que se pasaba al clasificador.

Efecto observado:

- El tiempo de entrenamiento smoke pasó de unos 22.0 s a 21.891 s.
- Conclusión: la limpieza es correcta a nivel lógico, pero su impacto en tiempo es despreciable en el dataset smoke.

Riesgo:

- Bajo. Solo elimina trabajo muerto.

### 2. Caché de reglas de renombrado de canales

Cambio:

- Se añadió una caché en proceso para las reglas de renombrado cargadas desde `channel_table.csv`.
- Se sustituyeron las llamadas repetidas a `load_rename_rules(os.path.abspath(csv_path))` por una consulta a la caché.

Motivo:

- `extract_physiological_features` estaba cargando y parseando el mismo CSV para cada registro.

Efecto observado:

- El tiempo smoke medido en la siguiente ejecución fue 22.040 s.
- Conclusión: la optimización es correcta, pero no ataca un cuello de botella relevante en smoke.

Riesgo:

- Bajo. El comportamiento no cambia salvo por reutilizar reglas ya parseadas.

### 3. Caché de demographics y etiquetas para entrenamiento

Cambio:

- Se añadió una lectura única de `demographics.csv` al inicio de `train_model`.
- Se construyeron:
  - una caché de demographics indexada por `(patient_id, session_id)`;
  - una caché de diagnósticos indexada por `patient_id`.
- Se reemplazaron las llamadas por registro a `load_demographics(...)` y `load_diagnoses(...)` durante entrenamiento.

Motivo:

- El bucle original de entrenamiento releía el mismo CSV para cada registro.

Efecto observado:

- El tiempo smoke bajó a 20.837 s.
- Conclusión: es una mejora real, aunque moderada.

Riesgo:

- Bajo a medio.
- Asume que las etiquetas de entrenamiento son estables a nivel de paciente cuando se cachean por `patient_id`, igual que hacía el comportamiento original de `load_diagnoses(...)`.

### 4. Paralelización de la extracción de features en entrenamiento

Cambio:

- Se añadió procesamiento paralelo por registro con `ThreadPoolExecutor` dentro de `train_model`.
- Se movió la lógica de extracción por registro a `process_training_record(...)`.
- Se limitó el número de workers con:

```python
MAX_TRAIN_WORKERS = max(1, min(4, os.cpu_count() or 1))
```

Motivo:

- Cada registro de entrenamiento se procesa de forma independiente.
- El pipeline mezcla lecturas de archivos EDF y trabajo con NumPy, así que un pool pequeño de hilos puede reducir el tiempo total.

Efecto observado:

- El tiempo smoke bajó a 9.578 s en la primera ejecución tras paralelizar.
- Las ejecuciones de seguimiento midieron 9.762 s y 9.655 s.
- Conclusión: esta es la optimización dominante.

Riesgo:

- Medio.
- El acceso paralelo a archivos puede comportarse distinto en discos más lentos o en una infraestructura más limitada del Challenge.

## Plan De Rollback

Si la submission se comporta distinto en el entorno del Challenge, revertir en este orden:

1. Eliminar la extracción con hilos y restaurar el bucle secuencial original en `train_model`.
2. Eliminar las cachés de metadata de entrenamiento y volver a `load_demographics(...)` / `load_diagnoses(...)`.
3. Eliminar la caché de reglas de renombrado y volver a las llamadas directas a `load_rename_rules(...)`.
4. Rehabilitar la carga de anotaciones humanas solo si el vector de entrenamiento se modifica explícitamente para usar esas features.

Este orden de rollback elimina primero la optimización de mayor riesgo y deja para el final los cambios de comportamiento más pequeños.

## Archivos Modificados

- `team_code.py`