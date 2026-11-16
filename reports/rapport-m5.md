# Lab 5 — Déploiement du modèle de maintenance prédictive

## 1. Résumé
Le modèle ONNX de maintenance prédictive est servi par FastAPI dans le cluster
kind. KServe déploie le service, Knative ajuste le nombre de pods et la passerelle
Kourier reçoit les requêtes sur le port 30080.

## 2. Tests de charge
Locust a envoyé des requêtes POST /predict avec des lignes de test AI4I.
Les latences sont exprimées en millisecondes.

| Essai | Utilisateurs | Requêtes/s | P50 (ms) | P95 (ms) | P99 (ms) | Échecs |
|---|---:|---:|---:|---:|---:|---:|
| run_100 | 100 | 60.86 | 1400 | 1900 | 2900 | 1 |
| run_500 | 500 | 50.36 | 9000 | 12000 | 46000 | 1 |
| run_1000 | 1000 | 70.40 | 9200 | 39000 | 69000 | 2 |

Les échecs observés étaient des erreurs HTTP 502. Le P95 est passé de 1,9 s
à 39 s entre 100 et 1 000 utilisateurs. Les chiffres du test à 500 utilisateurs
proviennent du CSV. Ceux des deux autres essais proviennent des résumés Locust.

## 3. Autoscaling
La révision 00002 est descendue à zéro pod en l'absence de trafic.
L'appel à froid a répondu en 2,552 s, puis l'appel à chaud en 2,570 ms.
Les deux appels ont répondu avec HTTP 200. Le pic de pods, le CPU moyen et
le temps de montée en charge restent à renseigner à partir du suivi Knative.
Le délai exact de retour à zéro n'a pas été chronométré.

## 4. Analyse du déploiement canary
Sur 200 appels à /metadata, la v1 a répondu 178 fois (89 %) et la v2 22 fois
(11 %), ce qui est cohérent avec la configuration 90/10. Après promotion,
l'adresse principale a renvoyé la version d'API 2.0.0 et la version de modèle 1.

## 5. Pistes d'amélioration
Rejouer les tests en relevant le nombre de pods et leur CPU pour mieux
expliquer la hausse de latence. Comparer ensuite les réglages d'autoscaling
sur le nombre de requêtes simultanées avec les réglages actuels fondés sur le débit.
