# Promotion Board — piloto EXP-017 vs EXP-019

Comparación pareada con inferencia real y métrica oficial sobre el protocolo congelado `biohub-legacy-seed42-holdout-common12-v1`.

- Nodo: A1558, RTX 4090.
- EXP-017: Wide Anisotropic U-Net 64ch, threshold congelado `0.035`.
- EXP-019: Wide ResUNet3D 64ch, threshold congelado `0.050`.
- TTA: identidad + flip X.
- Tracker común: Hungarian + gap closing, sin mitosis.
- Matching oficial: `7 µm`.

Los subdirectorios `remote_results/EXP-017` y `remote_results/EXP-019` contienen los JSON crudos descargados de la remota. `board/` contiene el ranking recomputado y la comparación bootstrap pareada.

## Resultado

- EXP-017: score `0.8098`, IC 95% `[0.7431, 0.8727]`, runtime `421.0 s`.
- EXP-019: score `0.7783`, IC 95% `[0.7005, 0.8494]`, runtime `524.8 s`.
- Delta pareado: `+0.0315`, IC 95% `[+0.0046, +0.0592]`, `P(EXP-017 > EXP-019)=98.94 %`.
- Decisión piloto: EXP-017 `SUBMIT_CANDIDATE`; EXP-019 `HOLD`.

La infraestructura se considera válida para seguir ampliándola. El protocolo `common12` conserva limitaciones y no se declara todavía métrica global.
