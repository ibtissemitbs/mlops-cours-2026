
## Lab 1 — Maintenance prédictive AI4I

### Données et protocole
- Dataset synthétique AI4I 2020, téléchargé depuis UCI.
- Cible : Machine failure, classe positive = 1.
- Variables : températures, vitesse, couple, usure et Type.
- Les identifiants et les indicateurs TWF/HDF/PWF/OSF/RNF
  sont exclus des variables explicatives.
- Répartition stratifiée : 6400 train, 1600 validation, 2000 test.
- Sélection des configurations sur la ROC AUC de validation.
- Les deux premiers essais 80/20 sont préliminaires :
  leur test a déjà été consulté avant la sélection finale.

### Résultats
- 22 configurations classiques et 3 architectures MLP comparées.
- Modèle retenu : Gradient Boosting, 100 arbres, profondeur 3,
  learning_rate=0.1, random_state=42.
- Seuil de décision : 1/6.
- Test : ROC AUC 0.9638, average precision 0.7890,
  précision 0.6207, rappel 0.7941, F1 0.6968.
- Matrice de confusion : TN=1899, FP=33, FN=14, TP=54.
- Coût pédagogique : 5*FN + FP = 103, sans unité monétaire.
- Registry scoring-model : v1 Gradient Boosting @champion,
  v2 Random Forest, v3 MLP à une couche @challenger.

### Environnement et services
- Prérequis : environnement du Lab 0 et cluster kind-mlops-lab.
- Installer les dépendances avec uv sync.
- Activer : source .venv/bin/activate
- Charger la configuration locale :
  set -a; source .env; set +a
- Le fichier .env est local et ne doit pas être ajouté à Git.
- Garder les port-forwards actifs dans des terminaux séparés :
  kubectl --context kind-mlops-lab -n storage port-forward svc/s3 8333:8333 8888:8888
  kubectl --context kind-mlops-lab -n mlflow port-forward svc/mlflow 5000:5000

### Reproduire les données et le modèle
- Utiliser la branche lab/m1-versioning.
- Remote DVC : s3://dvc-store/ai4i, endpoint http://localhost:8333.
- Configurer les identifiants du stockage dans .dvc/config.local
  avec dvc remote modify --local ; ne pas les commiter.
- Même cluster, stockage déjà rempli :
  dvc pull
  dvc repro
  dvc metrics show
- Autre cluster, stockage vide : reconstruire avec dvc repro,
  puis alimenter son stockage avec dvc push.
- Les expériences et alias MLflow ne sont pas restaurés par dvc pull :
  ils appartiennent au serveur MLflow de chaque cluster.
- Pour recréer les expériences : lancer les entraînements MLP et
  src.grid_search, puis enregistrer les candidats avec src.promote.

### Feast
- Les définitions se trouvent dans features/.
- Depuis features/, exécuter feast apply puis :
  feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
- Le registre et le stockage SQLite features/data/ sont locaux,
  ignorés par Git et reconstruits avec Feast.
- Les accès offline et online ont été vérifiés :
  les six variables correspondent aux données préparées.
- L'horodatage est technique et fixe, pas une date réelle de mesure.

### Vérifications réalisées
- Pipeline complet : download -> prepare -> train -> evaluate.
- Sans changement, dvc repro ignore les quatre étapes.
- Restauration validée dans une copie propre sur le même poste,
  avec le même environnement Python et le même cluster.
- Après recalcul forcé : différences de métriques nulles.
- Seul le hash du modèle dans dvc.lock change avec les métadonnées MLflow.

### Incident rencontré
- Le téléchargement automatique des artefacts MLflow a échoué.
- Le modèle sélectionné a pu être récupéré avec curl via le proxy
  d'artefacts, puis chargé localement pour le test.
- Le pipeline DVC évalue directement models/model, sans téléchargement.
