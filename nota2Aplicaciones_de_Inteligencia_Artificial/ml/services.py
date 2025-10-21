# ml/services.py
import json
import joblib
import pathlib
import pandas as pd   # <-- agrega esto

BASE = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"

INS_MODEL_PATH = MODELS_DIR / "insurance_model.pkl"
INS_PIPE_PATH  = MODELS_DIR / "insurance_pipeline.pkl"
DIA_MODEL_PATH = MODELS_DIR / "diabetes_model.pkl"
DIA_PIPE_PATH  = MODELS_DIR / "diabetes_pipeline.pkl"
THR_PATH       = MODELS_DIR / "threshold.json"

_artifacts = None

def _load_artifacts():
    global _artifacts
    if _artifacts is not None:
        return _artifacts
    missing = [p for p in [INS_MODEL_PATH, INS_PIPE_PATH, DIA_MODEL_PATH, DIA_PIPE_PATH, THR_PATH] if not p.exists()]
    if missing:
        files = "\n - ".join(str(m) for m in missing)
        raise FileNotFoundError(
            "Faltan artefactos de modelo.\n"
            f"No encontrados:\n - {files}\n"
            "Ejecuta: python manage.py train_all"
        )
    _artifacts = {
        "ins_model": joblib.load(INS_MODEL_PATH),
        "ins_pipe":  joblib.load(INS_PIPE_PATH),
        "dia_model": joblib.load(DIA_MODEL_PATH),
        "dia_pipe":  joblib.load(DIA_PIPE_PATH),
        "thr": float(json.load(open(THR_PATH, "r", encoding="utf-8"))["diabetes_threshold"]),
    }
    return _artifacts

def predict_insurance(payload: dict) -> float:
    art = _load_artifacts()
    import pandas as pd
    X_df = pd.DataFrame([payload])
    X = art["ins_pipe"].transform(X_df)
    y_usd = art["ins_model"].predict(X)[0]

    # Conversión: USD → CLP (valor aproximado del dólar)
    usd_to_clp = 970  # puedes actualizarlo si cambia el tipo de cambio
    y_clp_anual = y_usd * usd_to_clp

    # Convertir a costo mensual
    y_clp_mensual = y_clp_anual / 12

    return float(y_clp_mensual)


def predict_diabetes(payload: dict) -> dict:
    art = _load_artifacts()
    X_df = pd.DataFrame([payload])           # <-- DataFrame (1 fila)
    X = art["dia_pipe"].transform(X_df)
    p = float(art["dia_model"].predict_proba(X)[0, 1])
    pred = int(p >= art["thr"])
    return {"prob": p, "pred": pred, "threshold": art["thr"]}
