# Benchmark — ai4i-predictive-maintenance

Modèle : version 1

1000 appels par scénario, après 10 appels de chauffe.

Python : 3.12.3 ; architecture : x86_64.

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline | 3.4683 | 5.1379 | 6.2115 |
| onnx_python | 0.0345 | 0.0441 | 0.0597 |
| onnx_batch_100 | 0.0992 | 0.1567 | 0.2346 |

Accélération ONNX en P50 sur une ligne : x100.5.

Le scénario batch mesure le temps total pour 100 lignes.
Les mesures excluent le réseau HTTP et le chargement des modèles. Les entrées ONNX sont préparées à l'avance ; le pipeline scikit-learn reçoit un DataFrame.
