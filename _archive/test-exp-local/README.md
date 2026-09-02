# test-exp-local

Evaluador canónico para decidir qué experimentos Biohub promover a Kaggle.
Todo el código, protocolos y tests propios del evaluador viven en esta carpeta.
Los checkpoints, datasets y artefactos de resultados permanecen fuera porque
son entradas/salidas, no código del evaluador.

## Componentes

- `evaluator.py`: inferencia real de checkpoints Anisotropic U-Net, Wide U-Net,
  Wide ResUNet y StarDist 3D, seguida por la métrica oficial.
- `validate_evaluation.py`: valida un `evaluation.json` y recalcula sus agregados.
- `promotion_board.py`: bootstrap pareado, gates y ranking conservador.
- `build_board.py`: CLI que produce el tablero JSON y Markdown.
- `protocols/`: conjuntos de evaluación congelados.
- `tests/`: pruebas CPU del agregador y sus restricciones de comparabilidad.
- `run_legacy_four_model_benchmark.sh`: reproduce en secuencia el benchmark
  EXP-015/017/019/022 y construye el tablero de cuatro modelos.

## Evaluación reproducible

El protocolo piloto congelado compara EXP-017 y EXP-019 sobre las mismas 12
muestras que no participaron en su entrenamiento histórico. Los hiperparámetros
de inferencia deben conservarse para que la comparación sea válida.

```bash
python test-exp-local/evaluator.py \
  --experiment-id EXP-017 \
  --architecture wide_unet3d \
  --checkpoint checkpoints/exp017_wide_unet3d_90_10.pt \
  --project-root . \
  --data-dir data/train \
  --protocol test-exp-local/protocols/legacy_exp017_exp019_pilot.json \
  --output-dir experiments/test-exp-local/EXP-017 \
  --threshold 0.035
```

Para EXP-019 se usa `wide_resunet3d`, su checkpoint y umbral `0.050`.
EXP-015 usa `anisotropic_unet3d`, 32 canales, umbral `0.035` y TTA `flip_x`.
EXP-022 usa `stardist3d`, 32 canales, umbral `0.035` y TTA `identity`.

## Validación y ranking

```bash
python test-exp-local/validate_evaluation.py \
  --evaluation-json experiments/test-exp-local/EXP-017/evaluation.json

python test-exp-local/build_board.py \
  experiments/test-exp-local/EXP-017/evaluation.json \
  experiments/test-exp-local/EXP-019/evaluation.json \
  --output-dir experiments/test-exp-local/board \
  --bootstrap-iterations 10000
```

La decisión principal es el límite inferior del IC bootstrap 95% del score
oficial. Estado completo, cobertura total, runtime y ausencia de muestras
omitidas son gates duros. No se inventa un score combinando diagnósticos.

## Limitaciones del piloto legacy

- Sólo incluye 12 de las 20 muestras del holdout histórico.
- El split original fue aleatorio por volumen, no separado por embrión.
- Los umbrales históricos se eligieron sobre la misma familia de holdout.
- Dos modelos no bastan para validar que la métrica prediga universalmente Kaggle.

Por eso el resultado sirve como prueba retrospectiva del evaluador y como
criterio interno comparable, no como estimador garantizado del leaderboard.

## Benchmark retrospectivo de cuatro modelos

Los cuatro modelos fueron entrenados con el split 90/10 de seed 42 y poseen
CV histórico registrado: EXP-015 `0.8096`, EXP-017 `0.9010`, EXP-019 `0.8867`
y EXP-022 `0.8601`. Para ejecutarlos secuencialmente:

```bash
PYTHON_BIN=/ruta/al/entorno/bin/python \
  bash test-exp-local/run_legacy_four_model_benchmark.sh /ruta/a/Biohub
```
