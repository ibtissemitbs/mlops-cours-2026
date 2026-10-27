# Rapport comparatif — Kubeflow Pipelines vs Airflow (Lab 2)

**Rédaction :** Ibtissem Moussa
**Date :** 27/10/2026
**Machine :** Intel Core Ultra 7 155H, 22 processeurs logiques visibles, 15 Gio de RAM disponibles pour WSL2 Ubuntu sous Windows 10.
**Dataset :** AI4I 2020, maintenance prédictive.

## 1. Contexte

Ce lab compare deux orchestrateurs pour un même pipeline de maintenance prédictive : Kubeflow Pipelines et Apache Airflow. Le traitement comporte six étapes : téléchargement, validation des données, préparation, entraînement, évaluation et enregistrement du modèle dans MLflow. Les données sont réparties en 6 400 observations pour l’entraînement, 1 600 pour la validation et 2 000 pour le test.

Les deux solutions réutilisent les modules Python du projet et l’image `localhost:5001/scoring-pipeline:1.0.0`. Les tâches s’exécutent dans le cluster Kubernetes kind à trois nœuds. KFP est installé dans Kubernetes ; les services Airflow fonctionnent dans Docker Compose et lancent leurs tâches avec KubernetesPodOperator. MLflow centralise le suivi des expériences et les versions de modèles.

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation de zéro | Non mesurée | Non mesurée |
| Temps jusqu’au premier succès | Non mesuré | Non mesuré |
| Durée d’un run complet | Non relevée | Environ 1 min 21 s |
| Relance identique | Réussie, trois premières étapes en cache ; durée non relevée | Non chronométrée |
| Modification et observation du résultat | Non chronométrées | Non chronométrées |
| Lignes des fichiers d’orchestration | 361 | 90 |
| Mémoire instantanée | Non isolée dans les nœuds kind | Environ 2,10 Gio, PostgreSQL et Redis inclus |

Le comptage KFP inclut les 236 lignes des composants et les 125 lignes du pipeline. Airflow comporte 90 lignes dans son DAG, mais utilise également le module partagé `pipelines/stage.py`.

Les exécutions ont réussi avec les deux orchestrateurs. Un modèle enregistré par Airflow apparaît dans MLflow avec le tag `launched_by=airflow`. Les contrôles automatiques techniques du lab passent tous : 18 vérifications sur 19 avant rédaction du rapport.

## 3. Analyse qualitative

### Developer experience

KFP représente explicitement les entrées, sorties et dépendances des composants. Le pipeline Python est compilé en YAML avant son import dans l’interface. Airflow utilise un DAG Python et réemploie les scripts existants à travers un exécuteur partagé. Dans cette implémentation, son fichier d’orchestration est plus court.

### Scaling

Les deux variantes exécutent leurs tâches dans Kubernetes. Le lab permet de définir des ressources mémoire, des délais et des reprises. Cependant, les essais sur une seule machine ne permettent pas de comparer leur capacité à traiter plusieurs pipelines simultanément.

### Debugging

Les interfaces permettent de suivre chaque étape et de consulter ses journaux. Le diagnostic KFP a notamment révélé un exécutable launcher vide dans une image décompressée sur un nœud. Une image réparée a permis de reprendre les exécutions. Avec Airflow, les commandes de diagnostic ont confirmé le chargement du DAG, son activation et l’absence d’erreurs d’import.

## 4. Recommandation

Pour ce projet, KFP convient lorsque la priorité est le suivi des composants ML, des artefacts et du cache. Airflow constitue une option adaptée pour intégrer ce traitement dans un ensemble de workflows de données. Le choix doit aussi tenir compte des outils déjà maîtrisés et de l’infrastructure disponible. Dans les deux cas, MLflow conserve son rôle de suivi et de registre des modèles.

## 5. Limites de la comparaison

Cette comparaison repose sur un petit dataset et quelques exécutions locales. Plusieurs durées n’ont pas été chronométrées. La consommation mémoire est une mesure ponctuelle, sans charge contrôlée. Celle de KFP n’a pas été séparée des autres services Kubernetes. Les différences de code reflètent aussi les abstractions choisies. Ces résultats ne permettent donc pas de désigner un orchestrateur globalement plus rapide ou plus économe.
