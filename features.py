import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def _augment_image(img: np.ndarray, n_extra: int, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Génère des variantes d'une image pour data augmentation.
    Retourne [original, aug1, aug2, ...] soit 1 + n_extra images.
    """
    out: List[np.ndarray] = [img.copy()]
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    for _ in range(n_extra):
        variant = img.copy().astype(np.float32)

        # 1. Rotation légère (-8° à +8°)
        angle = float(rng.uniform(-8, 8))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variant = cv2.warpAffine(variant, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 2. Luminosité / contraste
        alpha = float(rng.uniform(0.92, 1.08))
        beta = float(rng.uniform(-12, 12))
        variant = np.clip(alpha * variant + beta, 0, 255).astype(np.uint8)

        # 3. Bruit gaussien léger
        noise = rng.normal(0, 3, variant.shape).astype(np.float32)
        variant = np.clip(variant.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        out.append(variant)

    return out


def _auto_crop_to_ink(gray: np.ndarray) -> np.ndarray:
    """
    Recadre automatiquement l'image autour des pixels "encre" (texte),
    pour réduire l'impact du fond (table, marges, ombres).
    """
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image (2D).")

    # Binarisation pour détecter l'encre (texte sombre)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # On cherche les pixels sombres (encre) -> invert
    ink = 255 - th
    # Nettoyage léger
    kernel = np.ones((3, 3), np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.where(ink > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # marge de sécurité
    h, w = gray.shape[:2]
    pad_x = int(0.05 * (x1 - x0 + 1))
    pad_y = int(0.05 * (y1 - y0 + 1))
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)

    return gray[y0 : y1 + 1, x0 : x1 + 1]


def preprocess_grayscale(gray: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Prétraitement robuste pour l'écriture manuscrite:
    - recadrage auto sur le texte
    - amélioration contraste
    - resize final
    """
    gray = _auto_crop_to_ink(gray)
    # amélioration du contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (size, size))
    return gray


def load_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return preprocess_grayscale(img, size=256)


def load_grayscale_from_bytes(data: bytes, size: int = 256) -> np.ndarray:
    """
    Charge une image depuis des bytes (upload) avec le MÊME pipeline que l'entraînement.
    Utilise cv2 pour garantir une conversion grayscale identique à load_grayscale_image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Impossible de charger l'image (format invalide ou corrompu).")
    return preprocess_grayscale(img, size=size)


def lbp_features(
    image: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
    method: str = "uniform",
    normalize: bool = True,
    grid_x: int = 1,
    grid_y: int = 1,
) -> np.ndarray:
    lbp = local_binary_pattern(image, n_points, radius, method)

    if method == "uniform":
        n_bins = n_points + 2
    else:
        # Attention: 2**n_points devient énorme si n_points grand
        n_bins = int(lbp.max() + 1)

    h, w = lbp.shape[:2]
    grid_x = max(1, int(grid_x))
    grid_y = max(1, int(grid_y))
    cell_w = max(1, w // grid_x)
    cell_h = max(1, h // grid_y)

    feats = []
    for gy in range(grid_y):
        for gx in range(grid_x):
            x0 = gx * cell_w
            y0 = gy * cell_h
            x1 = w if gx == grid_x - 1 else (gx + 1) * cell_w
            y1 = h if gy == grid_y - 1 else (gy + 1) * cell_h
            cell = lbp[y0:y1, x0:x1]

            hist, _ = np.histogram(
                cell.ravel(), bins=n_bins, range=(0, n_bins), density=False
            )
            hist = hist.astype("float32")
            if normalize:
                s = hist.sum()
                if s > 0:
                    hist /= s
            feats.append(hist)

    return np.concatenate(feats, axis=0)


def extract_features_from_dataset(
    dataset_dir: str,
    radius: int = 3,
    n_points: int = 24,
    method: str = "uniform",
    grid_x: int = 1,
    grid_y: int = 1,
    augment: bool = True,
    n_augmented: int = 5,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parcourt un dataset structuré comme :

    dataset/
        writer1/
            img1.png
            img2.png
        writer2/
            img1.png
            img2.png

    Si augment=True, crée des variantes (rotation, luminosité, bruit) pour enrichir le dataset.
    Retourne X, y, class_names.
    """
    features: List[np.ndarray] = []
    labels: List[int] = []
    class_names: List[str] = []
    rng = np.random.default_rng(random_state)

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    writers = sorted(
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    )

    for label_idx, writer in enumerate(writers):
        writer_dir = os.path.join(dataset_dir, writer)
        class_names.append(writer)

        for fname in os.listdir(writer_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue

            fpath = os.path.join(writer_dir, fname)
            try:
                img = load_grayscale_image(fpath)
                images_to_use: List[np.ndarray] = (
                    _augment_image(img, n_augmented, rng) if augment else [img]
                )
                for aug_img in images_to_use:
                    feat = lbp_features(
                        aug_img,
                        radius=radius,
                        n_points=n_points,
                        method=method,
                        grid_x=grid_x,
                        grid_y=grid_y,
                    )
                    features.append(feat)
                    labels.append(label_idx)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    if not features:
        raise ValueError("No valid images found in dataset.")

    X = np.vstack(features)
    y = np.array(labels, dtype=np.int32)
    return X, y, class_names

