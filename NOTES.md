# Lab 0 — Setup

- Environnement : Ubuntu/WSL2 avec Docker Desktop.
- Cluster kind : trois nœuds Kubernetes Ready.
- SeaweedFS et MLflow déployés et accessibles.
- Test MLflow : run hello-mlops et artefact hello.txt enregistrés.
- Test DVC : fichier envoyé vers mlops-test/dvcstore.
- Validation avant push : 18/19 contrôles réussis.

## Corrections nécessaires

- Permissions d'exécution ajoutées aux scripts de création du cluster et de vérification.
- enableServiceLinks: false ajouté à SeaweedFS pour éviter le conflit S3_PORT.
- Filtres pkill précisés dans Makefile pour éviter l'arrêt du shell de make.
