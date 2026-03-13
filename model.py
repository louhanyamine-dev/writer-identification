from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class NoOpTransformer:
    """
    Transformer minimal pour représenter un pipeline "sans PCA"
    tout en gardant une interface .fit/.transform picklable.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NoOpTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


@dataclass
class WriterIdentificationModel:
    scaler: StandardScaler
    pca: Any  # PCA ou NoOpTransformer
    classifier: SVC
    class_names: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_pca)
        # Approximate probabilities from decision function when probas not available
        decision = self.classifier.decision_function(X_pca)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        # Softmax
        e = np.exp(decision - decision.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> str:
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        y_pred = self.classifier.predict(X_pca)
        idx = int(y_pred[0])
        return self.class_names[idx]


@dataclass
class WriterIdentificationBundle:
    """
    Bundle contenant deux modèles pour comparer :
    - avec PCA
    - sans PCA
    """

    with_pca: WriterIdentificationModel
    without_pca: WriterIdentificationModel

