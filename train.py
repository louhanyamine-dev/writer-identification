import argparse
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features_from_dataset
from model import NoOpTransformer, WriterIdentificationBundle, WriterIdentificationModel


def train_model(
    dataset_dir: str,
    model_path: str = "model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    n_components: Optional[int] = 50,
    lbp_radius: int = 3,
    lbp_points: int = 24,
    lbp_method: str = "uniform",
    lbp_grid_x: int = 4,
    lbp_grid_y: int = 4,
    augment: bool = True,
    n_augmented: int = 5,
) -> None:
    print(f"Loading dataset from: {dataset_dir}")
    if augment:
        print(f"Data augmentation: ON (x{1 + n_augmented} par image)")
    X, y, class_names = extract_features_from_dataset(
        dataset_dir,
        radius=lbp_radius,
        n_points=lbp_points,
        method=lbp_method,
        grid_x=lbp_grid_x,
        grid_y=lbp_grid_y,
        augment=augment,
        n_augmented=n_augmented,
        random_state=random_state,
    )

    print(f"Dataset loaded. Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(class_names)}")

    n_samples = X.shape[0]

    # Gestion des petits jeux de données : éviter les erreurs de stratification
    if n_samples < 10:
        print(
            "Small dataset detected (n < 10). "
            "Using all samples for both training and evaluation (no train/test split)."
        )
        X_train, X_test, y_train, y_test = X, X, y, y
        effective_test_size = None
    else:
        # S'assurer que le test contient au moins 1 échantillon par classe
        min_test = max(len(class_names), int(np.ceil(test_size * n_samples)))
        eff_test_size = min_test / n_samples
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eff_test_size, random_state=random_state, stratify=y
        )
        effective_test_size = test_size

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===== Modèle AVEC PCA =====
    # Ajuster n_components après le split: ne peut pas dépasser n_samples_train ni n_features
    if n_components is not None:
        max_comp = min(n_components, X_train_scaled.shape[0], X_train_scaled.shape[1])
        if max_comp <= 0:
            pca = PCA(random_state=random_state)
            n_components_effective = None
        else:
            pca = PCA(n_components=max_comp, random_state=random_state)
            n_components_effective = max_comp
    else:
        pca = PCA(random_state=random_state)
        n_components_effective = None

    print("Fitting PCA (with PCA model)...")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA reduced dimensionality to: {X_train_pca.shape[1]} components")

    clf_with_pca = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )

    print("Training SVM (with PCA)...")
    clf_with_pca.fit(X_train_pca, y_train)

    print("Evaluating (with PCA) on test set...")
    y_pred_with = clf_with_pca.predict(X_test_pca)
    acc_with = float(accuracy_score(y_test, y_pred_with))
    print(f"Accuracy (with PCA): {acc_with:.4f}")
    print(classification_report(y_test, y_pred_with, target_names=class_names))

    model_with_pca = WriterIdentificationModel(
        scaler=scaler,
        pca=pca,
        classifier=clf_with_pca,
        class_names=class_names,
        config={
            "use_pca": True,
            "n_components": n_components_effective,
            "test_size": effective_test_size,
            "lbp_radius": lbp_radius,
            "lbp_points": lbp_points,
            "lbp_method": lbp_method,
            "lbp_grid_x": lbp_grid_x,
            "lbp_grid_y": lbp_grid_y,
        },
        metrics={"accuracy": acc_with},
    )

    # ===== Modèle SANS PCA =====
    noop = NoOpTransformer().fit(X_train_scaled, y_train)
    X_train_no = noop.transform(X_train_scaled)
    X_test_no = noop.transform(X_test_scaled)

    clf_no_pca = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
    print("Training SVM (without PCA)...")
    clf_no_pca.fit(X_train_no, y_train)

    print("Evaluating (without PCA) on test set...")
    y_pred_no = clf_no_pca.predict(X_test_no)
    acc_no = float(accuracy_score(y_test, y_pred_no))
    print(f"Accuracy (without PCA): {acc_no:.4f}")
    print(classification_report(y_test, y_pred_no, target_names=class_names))

    model_without_pca = WriterIdentificationModel(
        scaler=scaler,
        pca=noop,
        classifier=clf_no_pca,
        class_names=class_names,
        config={
            "use_pca": False,
            "n_components": None,
            "test_size": effective_test_size,
            "lbp_radius": lbp_radius,
            "lbp_points": lbp_points,
            "lbp_method": lbp_method,
            "lbp_grid_x": lbp_grid_x,
            "lbp_grid_y": lbp_grid_y,
        },
        metrics={"accuracy": acc_no},
    )

    bundle = WriterIdentificationBundle(with_pca=model_with_pca, without_pca=model_without_pca)

    print(f"Saving model to: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a writer identification model using LBP + PCA + SVM."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model.pkl",
        help="Output path for trained model pickle (default: model.pkl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size fraction (default: 0.2)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of PCA components (default: 50)",
    )
    parser.add_argument(
        "--lbp-radius",
        type=int,
        default=3,
        help="LBP radius R (default: 3)",
    )
    parser.add_argument(
        "--lbp-points",
        type=int,
        default=24,
        help="LBP points P (default: 24)",
    )
    parser.add_argument(
        "--lbp-method",
        type=str,
        default="uniform",
        help="LBP method (default: uniform)",
    )
    parser.add_argument(
        "--lbp-grid-x",
        type=int,
        default=4,
        help="LBP grid blocks along X (default: 4)",
    )
    parser.add_argument(
        "--lbp-grid-y",
        type=int,
        default=4,
        help="LBP grid blocks along Y (default: 4)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Désactiver la data augmentation",
    )
    parser.add_argument(
        "--n-augmented",
        type=int,
        default=5,
        help="Nombre de variantes par image (default: 5, total = 6x par image)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        raise SystemExit(f"Dataset directory does not exist: {args.dataset}")

    train_model(
        dataset_dir=args.dataset,
        model_path=args.model_path,
        test_size=args.test_size,
        n_components=args.n_components,
        lbp_radius=args.lbp_radius,
        lbp_points=args.lbp_points,
        lbp_method=args.lbp_method,
        lbp_grid_x=args.lbp_grid_x,
        lbp_grid_y=args.lbp_grid_y,
        augment=not args.no_augment,
        n_augmented=args.n_augmented,
    )


if __name__ == "__main__":
    main()

