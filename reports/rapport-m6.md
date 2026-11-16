# Rapport — Lab 6 CI/CD/CT de bout en bout

Projet de maintenance prédictive sur AI4I. Dépôt `ibtissemitbs/mlops-cours-2026`.
Modèle final `scoring-model` v17, servi par l’image `scoring-api:2.1.8`.

## 1. Architecture

Evidently compare les données récentes aux données de référence. Il enregistre le rapport et le lot dans MLflow. Le client de détection envoie une alerte au webhook, qui démarre un pipeline Kubeflow si la dérive dépasse le seuil configuré et si le délai minimal est écoulé.

Kubeflow récupère le lot depuis MLflow, le valide avec Great Expectations, réentraîne le modèle et compare le candidat au champion. Si le candidat passe les contrôles, Kubeflow met à jour l’alias `champion` dans MLflow.

Le webhook exporte le champion en ONNX et pousse un commit sur `main`. GitHub Actions construit puis publie l’image dans GHCR et met à jour le manifeste de staging dans Git. Argo CD lit les manifestes Git et les applique dans Kubernetes. KServe sert les révisions en staging et en production. Le webhook vérifie les prédictions du canary avant de pousser le commit de promotion ou de retour arrière.

## 2. Temps du cycle

Horaires du 16 novembre 2026, en UTC+1. Run Kubeflow `c60a42ab-2313-4a03-940d-a2443e091c56`.

| Étape | Heure | Durée observée |
|---|---:|---:|
| Détection Evidently | 22:55:01 | — |
| Création du run après renvoi de l’alerte | 23:00:29 | 5 min 28 s depuis la détection |
| Démarrage du pipeline | 23:00:30 | — |
| Fin du pipeline | 23:04:06 | 3 min 37 s affichées par Kubeflow |
| Commit du modèle `8970f17` | 23:04:16 | 10 s après la fin du pipeline |
| Commit de staging `adbb746` | 23:06:35 | 2 min 19 s après le commit du modèle |
| Commit du canary `d3fb13d` | 23:07:18 | 43 s après le commit de staging |
| Commit de promotion `fcb965b` | 23:10:28 | 3 min 10 s après le commit du canary |

Le cycle a pris 9 min 59 s de la création du run au commit de promotion et 15 min 27 s depuis la détection initiale, attente et renvoi compris. Le principal délai avant le démarrage était lié au délai minimal entre deux réentraînements. Une alerte envoyée après ce délai évite cette attente dans la démonstration. Le test de 200 requêtes a confirmé la mise en service après le commit de promotion. L’heure exacte de cette vérification n’a pas été relevée.

## 3. Gates et décisions

Le lot de 10 000 lignes a passé les 25 contrôles Great Expectations. Evidently a détecté une dérive sur 2 variables sur 6, les températures de l’air et du procédé, soit 33,33 % pour un seuil de déclenchement de 10 %.

| Modèle | Résultat |
|---|---|
| v15, scénario à +3 K | Promue avec une ROC AUC de 0,9653 et un coût métier de 144, contre 0,9286 et 246 pour la v1 sur le même jeu de test. |
| v16, scénario à +4 K | Refusée malgré une ROC AUC de 0,9464. Son coût métier de 190 dépassait celui de la v15, qui était de 174 sur ce lot. |
| v17, scénario à +3 K rejoué | Promue par la gate, enregistrée comme `champion` dans MLflow, puis publiée sous le tag `2.1.8`. |

Le canary de la v17 a démarré à 10 %. Après ses vérifications de prédiction, le bot a poussé la promotion à 100 %. Argo CD a affiché `Synced` et `Healthy`, et KServe a affecté 100 % du trafic à la révision `scoring-model-predictor-00006`. Les 200 requêtes sur `/version` ont toutes renvoyé le tag `2.1.8` et le modèle `17`.

## 4. Points d’attention

Le webhook de démonstration écoute sur localhost au port 8096 et exige un jeton chargé depuis `.env`. Il impose un délai de 600 secondes entre deux réentraînements. Le push du modèle a demandé une authentification GitHub dans le terminal. La publication, le canary et la décision de promotion sont pilotés par le bot.

Le retour arrière se pilote par un commit GitOps. KServe redirige le trafic vers la révision précédente lorsque le canary revient à 0 %. L’alias `champion` de MLflow doit également refléter la version retenue.

## 5. Différences avec la production

En production, utiliser des secrets managés et une identité dédiée au bot avec des droits limités. Séparer les environnements, journaliser les décisions et encadrer la promotion par une revue humaine selon le risque métier. Compléter les tests de prédiction par un suivi des erreurs, de la latence et de la qualité métier sur des données récentes et étiquetées.

## 6. Preuves de la démonstration

- Kubeflow, onglet Detail du run `c60a42ab-2313-4a03-940d-a2443e091c56`.
- GitHub Actions, publication déclenchée par le commit `8970f17`.
- Argo CD, application `scoring-prod` synchronisée sur `fcb965b`.
- KServe, révision `scoring-model-predictor-00006` à 100 % et sortie du test des 200 requêtes.
