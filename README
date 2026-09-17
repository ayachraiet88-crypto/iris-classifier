#  Iris Flower Classifier — Classification d'espèces de fleurs

Application de Machine Learning classifiant une fleur d'Iris parmi 3 espèces (Setosa, Versicolor, Virginica) à partir de 4 mesures physiques. Projet basé sur le dataset Iris classique, avec sélection de variables via LassoCV, optimisation d'hyperparamètres via GridSearchCV, et déploiement dans une interface interactive Gradio.

##  Objectif du projet

Construire un classifieur simple mais rigoureux (sélection de features justifiée, hyperparamètres optimisés) sur un dataset de référence en Machine Learning, avec une interface interactive permettant de tester des prédictions en temps réel.

##  Dataset

- **Source** : Iris dataset (intégré à scikit-learn, `sklearn.datasets.load_iris`)
- **150 fleurs**, 4 mesures (longueur/largeur des sépales et pétales), 3 classes parfaitement équilibrées (50 exemples chacune) : Setosa, Versicolor, Virginica

## Stack technique

- **Langage** : Python
- **ML** : scikit-learn (LassoCV, KNeighborsClassifier, GridSearchCV, StandardScaler)
- **Déploiement** : Gradio
- **Sérialisation** : joblib

## Pipeline du projet

1. **Chargement** des données via l'API scikit-learn (`load_iris`)
2. **Split stratifié** (`stratify=y`) pour préserver l'équilibre parfait des 3 classes
3. **Standardisation** des features (`StandardScaler`)
4. **Sélection de variables via LassoCV** : identification des features réellement discriminantes
5. **Classification via K-Nearest Neighbors (KNN)**, avec optimisation des hyperparamètres (`n_neighbors`, `weights`) par `GridSearchCV`
6. **Évaluation** via `classification_report`
7. **Déploiement** dans une interface Gradio avec sliders interactifs

##  Résultats

### Sélection de variables (LassoCV)

| Variable | Coefficient |
|---|---|
| sepal length (cm) | -0.0 |
| sepal width (cm) | -0.039 |
| petal length (cm) | 0.258 |
| **petal width (cm)** | **0.512** |

Les dimensions des **pétales** (longueur et largeur) sont nettement plus discriminantes que celles des sépales pour distinguer les espèces — la longueur des sépales a même un coefficient nul, écartée par la régularisation Lasso.

### Optimisation KNN (GridSearchCV)

- Meilleurs paramètres : `n_neighbors=5`, `weights='uniform'`
- Meilleur score (validation croisée) : **0.967**

### Performance finale

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 0.83 | 1.00 | 0.91 |
| Virginica | 1.00 | 0.80 | 0.89 |
| **Accuracy globale** | | | **0.93** |

## Application Gradio

Interface interactive avec 4 sliders (longueur/largeur sépale et pétale), qui prédit l'espèce en temps réel avec les probabilités pour chacune des 3 classes.

### Lancer l'application

```bash
git clone <repo-url>
cd iris-classifier
pip install -r requirements.txt
python app_iris.py
```

## Structure du projet

```
iris-classifier/
├── app_iris.py           # Application Gradio
├── iris_flower.ipynb     # Notebook d'entraînement et d'analyse
├── iris_model.pkl        # Modèle KNN entraîné (via GridSearchCV)
├── iris_scaler.pkl       # StandardScaler fitté
└── README.md
```

## Points clés méthodologiques

- **LassoCV pour la sélection de variables** : plutôt que de garder toutes les features par défaut, une régularisation L1 identifie objectivement lesquelles sont réellement utiles (ici, les pétales dominent largement les sépales)
- **Standardisation avant KNN** : indispensable pour cet algorithme basé sur les distances entre points — sans normalisation, les variables à plus grande échelle domineraient artificiellement le calcul de distance
- **`stratify=y`** : garantit un split représentatif malgré le faible nombre d'exemples par classe (150 au total, 50 par classe)
- **GridSearchCV** : évite de choisir `n_neighbors` "à l'instinct", teste plusieurs valeurs systématiquement avec validation croisée

## Auteur
Aya — Projet réalisé dans le cadre de la préparation au PFE (Big Data & Data Analysis, ISAMM)
