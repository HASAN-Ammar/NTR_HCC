# HCC Recurrence and NTR Risk (RF)

This repo includes:
- Training script for RF models (Stage1: recurrence, Stage2: NTR | recurrence)
- Streamlit app for researcher-facing predictions

## Quick start (local)

1) Create models (run once):
```
D:/Publicatin/HCC/.venv/Scripts/python.exe D:/Publicatin/HCC/shiny_app/train_models.py
```

2) Run the Streamlit app:
```
streamlit run D:/Publicatin/HCC/streamlit_app/app.py
```

## Streamlit Community Cloud

1) Push this repo to GitHub.
2) In Streamlit Cloud, set:
- App file path: `streamlit_app/app.py`
- Python version: 3.11 (or compatible)
3) The app will install dependencies from `requirements.txt`.
4) Make sure these model files exist in the repo after training:
- `shiny_app/models/stage1_rf.pkl`
- `shiny_app/models/stage2_rf.pkl`
- `shiny_app/models/features.json`

## Notes
- The app expects input columns matching the training features in `features.json`.
- Output includes P(Recurrence), P(NTR | Recurrence), and P(NTR).
