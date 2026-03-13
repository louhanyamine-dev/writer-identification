import io
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from features import lbp_features, load_grayscale_from_bytes
from model import WriterIdentificationBundle, WriterIdentificationModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model(model_path: str = "model.pkl") -> Optional[Any]:
    try:
        # Toujours charger le modèle par rapport au dossier du fichier app.py
        abs_path = model_path
        if not os.path.isabs(model_path):
            abs_path = os.path.join(BASE_DIR, model_path)
        with open(abs_path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, (WriterIdentificationModel, WriterIdentificationBundle)):
            st.warning("Loaded object is not a compatible model (WriterIdentificationModel/Bundle).")
            return None
        return model
    except FileNotFoundError:
        st.error(
            "Model file not found. Please train the model first by running `python train.py --dataset dataset`."
        )
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_uploaded_image(bytes_data: bytes, model_config: dict) -> np.ndarray:
    """
    Même pipeline que l'entraînement : cv2 pour chargement + preprocessing + LBP.
    Utilise les paramètres LBP sauvegardés dans model.config.
    """
    img_np = load_grayscale_from_bytes(bytes_data, size=256)
    r = model_config.get("lbp_radius", 3)
    p = model_config.get("lbp_points", 24)
    gx = model_config.get("lbp_grid_x", 4)
    gy = model_config.get("lbp_grid_y", 4)
    feat = lbp_features(img_np, radius=r, n_points=p, grid_x=gx, grid_y=gy)
    return feat.reshape(1, -1)

def get_models_and_metrics(obj: Any) -> Tuple[Dict[str, WriterIdentificationModel], Dict[str, Any]]:
    if isinstance(obj, WriterIdentificationModel):
        return {"Model": obj}, {"Model": obj.metrics or {}}
    if isinstance(obj, WriterIdentificationBundle):
        return (
            {"Avec PCA": obj.with_pca, "Sans PCA": obj.without_pca},
            {"Avec PCA": obj.with_pca.metrics or {}, "Sans PCA": obj.without_pca.metrics or {}},
        )
    return {}, {}


def main():
    st.set_page_config(
        page_title="Writer Identification",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .main-header{background:linear-gradient(135deg,#1e3a5f 0%,#2d5a87 100%);color:#fff;padding:1.5rem 2rem;
    border-radius:12px;margin-bottom:2rem;box-shadow:0 4px 20px rgba(30,58,95,.25)}
    .main-header h1{margin:0;font-size:1.75rem;font-weight:700}
    .main-header p{margin:.5rem 0 0;opacity:.95;font-size:.95rem}
    .stSidebar{background:linear-gradient(180deg,#f1f5f9 0%,#e2e8f0 100%)}
    div[data-testid="stMetricValue"]{font-weight:600;color:#1e3a5f}
    .stButton>button{background:linear-gradient(135deg,#1e3a5f,#2d5a87)!important;color:#fff!important;
    font-weight:600!important;padding:.6rem 1.5rem!important;border-radius:8px!important;border:none!important}
    .stButton>button:hover{box-shadow:0 4px 12px rgba(30,58,95,.35)!important}
    div[data-testid="stFileUploader"]{background:#f8fafc;border:2px dashed #cbd5e1;border-radius:12px;padding:2rem}
    .stDataFrame{border-radius:10px;overflow:hidden}
    </style>
    <div class="main-header"><h1>✍️ Identification des Écrivains — Écriture Manuscrite</h1>
    <p>LBP + PCA + SVM — Prédiction de l'auteur à partir d'une image</p></div>
    """, unsafe_allow_html=True)

    loaded = load_model("model.pkl")
    if loaded is None:
        st.stop()

    models, metrics = get_models_and_metrics(loaded)
    model_choice = st.sidebar.selectbox("Mode de prédiction", list(models.keys()))
    model = models[model_choice]

    if metrics.get(model_choice, {}).get("accuracy") is not None:
        st.sidebar.metric("Accuracy (test)", f"{metrics[model_choice]['accuracy']*100:.2f}%")

    uploaded_files = st.file_uploader(
        "Uploader une ou plusieurs images manuscrites",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uf in enumerate(uploaded_files):
            try:
                image_preview = Image.open(io.BytesIO(uf.read()))
                cols[i % len(cols)].image(image_preview, caption=uf.name, use_container_width=True)
            except Exception:
                cols[i % len(cols)].warning(f"Impossible d'afficher: {uf.name}")

        if st.button("Predict Writer"):
            results = []
            progress = st.progress(0)
            unknown_threshold = 0.6  # seuil pour déclarer "Auteur inconnu"

            with st.spinner("Prédiction en cours..."):
                for idx, uf in enumerate(uploaded_files):
                    try:
                        # Re-lire le contenu (le buffer peut avoir été consommé pour l'aperçu)
                        uf.seek(0)
                        raw_bytes = uf.read()
                        config = getattr(model, "config", {}) or {}
                        X = preprocess_uploaded_image(raw_bytes, config)

                        proba = model.predict_proba(X)[0]
                        max_idx = int(np.argmax(proba))
                        max_conf = float(proba[max_idx])
                        predicted_writer = model.class_names[max_idx]

                        if max_conf < unknown_threshold:
                            display_writer = "Auteur inconnu"
                        else:
                            display_writer = predicted_writer

                        row = {
                            "image": uf.name,
                            "modele": model_choice,
                            "auteur_affiche": display_writer,
                            "auteur_max_proba": predicted_writer,
                            "confiance_max_%": round(max_conf * 100.0, 2),
                        }

                        # Ajouter toutes les probabilités par auteur pour analyse
                        for cls_name, p in zip(model.class_names, proba):
                            row[f"proba_{cls_name}_%"] = round(float(p) * 100.0, 2)

                        results.append(row)
                    except Exception as e:
                        results.append(
                            {
                                "image": uf.name,
                                "modele": model_choice,
                                "auteur_affiche": "Erreur",
                                "auteur_max_proba": None,
                                "confiance_max_%": None,
                                "details": str(e),
                            }
                        )
                    progress.progress(int(((idx + 1) / len(uploaded_files)) * 100))

            st.subheader("Résultats")
            st.caption(
                "La colonne 'auteur_affiche' tient compte d'un seuil de confiance. "
                "Si la confiance max est inférieure à 60%, l'auteur est affiché comme 'Auteur inconnu'."
            )
            st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()

