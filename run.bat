@echo off
cd /d "%~dp0"

echo === 1. Entrainement du modele (LBP simple, grille 1x1 = moins d'overfitting) ===
python train.py --dataset dataset --model-path model.pkl --n-components 50 --lbp-radius 3 --lbp-points 24 --lbp-grid-x 1 --lbp-grid-y 1

echo.
echo === 2. Lancement de l'application ===
streamlit run app.py

pause
