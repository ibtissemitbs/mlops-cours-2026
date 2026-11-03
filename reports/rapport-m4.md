# Rapport Lab 4 — Packaging du modèle AI4I

## 1. Comparatif des tailles et de la sécurité

Les trois images fournissent un service de prédiction de panne.
Les images naïve et optimisée utilisent FastAPI et ONNX.
L’image BentoML utilise le pipeline scikit-learn.

| Image | Taille affichée par docker images | HIGH | CRITICAL | Total |
|---|---:|---:|---:|---:|
| scoring-api:naive | 2,05 Go | 473 | 16 | 489 |
| scoring-api:optimized | 535 Mo | 45 | 0 | 45 |
| scoring-api:bentoml | 1,41 Go | 81 | 0 | 81 |

La mesure utilisée par le script check-m4 est de 125 Mo pour
l’image publiée. Le contrôle « taille < 500 Mo » est validé.
Cette mesure diffère de celle affichée par docker images.

Les résultats Trivy correspondent à la base de vulnérabilités
disponible au moment des scans.

## 2. Comparatif des latences

Modèle : Gradient Boosting, version 1 dans MLflow.
Chaque scénario comporte 1 000 appels après 10 appels de chauffe.

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| sklearn_pipeline | 3,4683 | 5,1379 | 6,2115 |
| onnx_python | 0,0345 | 0,0441 | 0,0597 |
| onnx_batch_100 | 0,0992 | 0,1567 | 0,2346 |

ONNX est environ 100,5 fois plus rapide en P50 sur une ligne
dans ce benchmark. Le scénario batch mesure le temps total
pour 100 lignes.

Ces mesures excluent le réseau HTTP et le chargement des modèles.
Les entrées ONNX sont préparées avant le chronométrage,
tandis que scikit-learn reçoit un DataFrame.

## 3. Interprétation des résultats

L’image optimisée utilise python:3.12-slim, une construction
multi-stage et une sélection des fichiers nécessaires.
Elle ne contient ni MLflow ni scikit-learn : ONNX Runtime
exécute le modèle déjà exporté. Le service fonctionne avec
l’utilisateur non-root 10001 et possède un HEALTHCHECK.

ONNX réduit le surcoût des appels Python et du prétraitement
du pipeline scikit-learn dans ce test.

La conversion a été vérifiée sur 2 000 lignes :
écart maximal de probabilité de 0,00000014 et aucune décision
différente au seuil de 0,1667.

Trivy détecte moins de vulnérabilités dans l’image optimisée.
Cela ne garantit pas l’absence de risque : les 45 signalements
HIGH restent à examiner.

ModelScan ignore le fichier ONNX : son message « No issues
found » ne constitue donc pas une validation de ce fichier.
La démonstration pickle détecte correctement un appel suspect
à system dans malicious.pkl. Ce fichier n’a pas été chargé.

L’inspection skops liste numpy.dtype et sklearn.tree._tree.Tree
comme types nécessitant une autorisation explicite.

## 4. Choix recommandé

Pour ce cas d’usage, je retiens l’image FastAPI + ONNX optimisée :
elle est plus légère, rapide dans le benchmark et présente moins
de vulnérabilités détectées que les deux autres images.

Elle contient le modèle et peut servir des prédictions sans
accéder à MLflow au démarrage.

L’image est publiée sous :
localhost:5001/scoring-api:1.0.0

Le pull, la route /health, les prédictions et le refus des
requêtes incomplètes avec HTTP 422 ont été vérifiés.
BentoML constitue une alternative qui automatise l’emballage
du service, avec davantage de dépendances dans cette expérience.

## 5. Pistes d’amélioration

- Examiner les vulnérabilités restantes et mettre à jour les
  composants concernés, puis refaire les tests et les scans.
- Épingler les images de base par digest pour améliorer
  la reproductibilité.
- Évaluer la quantification du modèle, en vérifiant son effet
  sur les probabilités et les décisions.
- Étudier une image distroless compatible avec les dépendances.
- Compléter le benchmark par un test de charge HTTP.
Le scan du fichier skops est incomplet : ModelScan rencontre
une incompatibilité avec NumPy lors de l’analyse des fichiers
.npy contenus dans l’archive (fonction _check_version absente).
Le message « No issues found » ne permet donc pas de conclure
à l’absence de problème. Le rapport d’erreurs est conservé
dans reports/modelscan_skops.txt.
