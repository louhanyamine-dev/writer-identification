# Identification des Écrivains à partir de l'Écriture Manuscrite

Application de reconnaissance d'auteur à partir d'images de textes manuscrits, utilisant **LBP** (Local Binary Pattern), **PCA** (Analyse en Composantes Principales) et **SVM** (Support Vector Machine).

---

## Objectifs

- Reconnaître l'auteur d'un texte manuscrit à partir d'une image
- Extraire les caractéristiques de texture avec LBP
- Réduire la dimensionnalité avec PCA
- Classifier les auteurs avec un SVM
- Fournir une interface utilisateur simple (Streamlit)

---

## Technologies utilisées

| Outil | Rôle |
|-------|------|
| **Python** | Langage principal |
| **OpenCV** | Prétraitement des images (grayscale, redimensionnement, CLAHE) |
| **scikit-image** | Extraction LBP |
| **scikit-learn** | PCA, SVM, StandardScaler, évaluation |
| **Streamlit** | Interface web |
| **Pillow** | Manipulation d'images (aperçu) |
| **NumPy** | Calculs numériques |

---

## Structure du projet

```
projetai/
├── app.py           # Interface Streamlit (upload, prédiction)
├── features.py      # Extraction LBP, prétraitement, augmentation
├── model.py         # Définition du modèle (WriterIdentificationModel)
├── train.py         # Script d'entraînement
├── model.pkl        # Modèle sauvegardé (généré après entraînement)
├── requirements.txt # Dépendances
├── run.bat          # Script pour entraîner + lancer l'app (Windows)
├── dataset/         # Images par auteur
│   ├── amine/
│   │   ├── image1.png
│   │   └── image2.jpeg
│   ├── hossin/
│   │   └── ...
│   └── soufian/
│       └── ...
└── README.md
```

---

## Structure du dataset

Chaque **auteur = un dossier** dans `dataset/` :

```
dataset/
   writer1/     ← nom de l'auteur
      img1.jpg
      img2.png
   writer2/
      img1.jpg
      ...
```

- Formats acceptés : PNG, JPG, JPEG, BMP, TIF, TIFF
- Chaque dossier = une classe (un auteur)
- Plus il y a d'images par auteur, meilleure sera la précision

---

## Installation

```bash
cd projetai
pip install -r requirements.txt
```

---

## Entraînement du modèle

```bash
python train.py --dataset dataset --model-path model.pkl
```

### Options d'entraînement

| Option | Défaut | Description |
|--------|--------|-------------|
| `--dataset` | `dataset` | Chemin du dossier dataset |
| `--model-path` | `model.pkl` | Fichier de sortie du modèle |
| `--n-components` | `50` | Nombre de composantes PCA |
| `--lbp-radius` | `3` | Rayon R du LBP |
| `--lbp-points` | `24` | Nombre de voisins P du LBP |
| `--lbp-grid-x` | `4` | Grille LBP (colonnes) |
| `--lbp-grid-y` | `4` | Grille LBP (lignes) |
| `--n-augmented` | `5` | Nombre de variantes par image (data augmentation) |
| `--no-augment` | - | Désactiver l'augmentation |

### Exemples

```bash
# Entraînement standard
python train.py --dataset dataset --model-path model.pkl

# Avec paramètres LBP personnalisés
python train.py --dataset dataset --lbp-radius 2 --lbp-points 16 --lbp-grid-x 2 --lbp-grid-y 2

# Sans augmentation
python train.py --dataset dataset --no-augment
```

---

## Lancer l'application

```bash
streamlit run app.py
```

Puis ouvrir `http://localhost:8501` dans le navigateur.

Sous Windows, vous pouvez aussi exécuter `run.bat` (entraînement + lancement).

---

## Pipeline technique

### 1. Prétraitement

- Conversion en niveaux de gris
- Recadrage automatique sur le texte (détection de l'encre)
- Amélioration du contraste (CLAHE)
- Redimensionnement à 256×256 pixels

### 2. Extraction des caractéristiques (LBP)

- **LBP** (Local Binary Pattern) : transforme chaque pixel en un motif binaire basé sur ses voisins
- Paramètres : **R** (rayon), **P** (nombre de voisins), méthode `uniform`
- Grille : découpe de l'image en blocs (ex. 4×4), histogramme LBP par bloc
- Résultat : vecteur de features (ex. 26×16 = 416 valeurs)

### 3. Réduction de dimension (PCA)

- **PCA** : conserve les directions de plus grande variance
- Réduit le vecteur (ex. 416 → 50 composantes)
- Diminue le sur-apprentissage et accélère le SVM

### 4. Classification (SVM)

- **SVM** (Support Vector Machine) avec noyau RBF
- Chaque auteur = une classe
- `class_weight="balanced"` pour gérer les classes déséquilibrées
- Sortie : nom de l'auteur + probabilités par classe

---

## Ajouter un nouvel auteur

1. Créer un dossier dans `dataset/` avec le nom de l'auteur (ex. `dataset/yassine/`)
2. Ajouter des images de son écriture
3. Ré-entraîner : `python train.py --dataset dataset --model-path model.pkl`

---

## Seuil de confiance

- Si la confiance maximale est **< 60%**, l'application affiche **"Auteur inconnu"**
- Sinon, elle affiche le nom de l'auteur prédit

---

## Comparaison Avec / Sans PCA

L'application permet de choisir :
- **Avec PCA** : réduction de dimension, souvent plus robuste
- **Sans PCA** : utilisation directe des features LBP, parfois plus précis sur petits jeux de données

---

## Data augmentation

Pour enrichir le dataset, chaque image est automatiquement augmentée :
- Rotation légère (-8° à +8°)
- Variation de luminosité/contraste
- Bruit gaussien léger

Par défaut : 6 variantes par image (1 originale + 5 augmentées).

---

## Limites

- Performances limitées avec très peu d'images (< 10 par auteur)
- Recommandé : au moins 10–20 images par auteur, prises dans des conditions similaires
- Le modèle ne reconnaît que les auteurs présents dans le dataset d'entraînement

---

## Auteur du projet

Projet : Identification des Écrivains (LBP / SVM / ACP) — G14 : Amine Louhany

---

## Licence

Usage éducatif / projet académique.
