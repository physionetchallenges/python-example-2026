# Dataset smoke (Modo desarrollo)

Entrenar con el dataset completo tarda aproximadamente 30–40 minutos con el modelo de ejemplo.

Para desarrollo utilizamos un dataset reducido (5 sujetos por defecto).

Este documento describe cuándo y por qué usar smoke.
Los comandos de ejecución están centralizados en `docs/04_run_script.md`.

---

## Qué incluye

- Muestra reducida del dataset (5 sujetos por defecto)
- Estructura compatible con el flujo oficial del proyecto
- Directorio de salida en `data/training_smoke/`
- `demographics.csv` filtrado para que solo incluya los registros copiados al smoke

## Para qué se usa

- Validar cambios de código rápidamente
- Detectar errores de integración antes del entrenamiento completo
- Iterar en modo desarrollo (smoke) sin esperar ciclos largos

## Artefactos asociados

- Entrenamiento smoke: `model_smoke/`
- Predicciones (inferencia) smoke: `outputs_smoke/`

## Relación con el flujo principal

El dataset smoke se crea al inicio del ciclo de desarrollo y se usa junto con `train-dev` y `run-dev`.
El orden detallado de ejecución está en `docs/04_run_script.md`.

## ¿Cuándo usar smoke?

- Desarrollo de nuevas funcionalidades
- Comprobación rápida de que el código no rompe
- Validación de cambios en `team_code.py`

Nunca usar smoke para evaluar rendimiento final.