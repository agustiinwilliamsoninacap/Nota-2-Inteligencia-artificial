import json
import joblib
import numpy as np
import pandas as pd
import pathlib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

BASE = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- utilidades ----------

def _read_csv_auto(path: pathlib.Path) -> pd.DataFrame:
    """Lee CSV detectando separador automáticamente y evitando problemas de encoding."""
    return pd.read_csv(path, sep=None, engine="python")

# ---------- INSURANCE (regresión) ----------

def train_insurance(df: pd.DataFrame):
    """
    Entrena un modelo lineal para charges.
    Tolera nombres alternativos del target (cost, expenses, price...) y valida columnas.
    """
    # Normaliza nombre del target si no viene como 'charges'
    if "charges" not in df.columns:
        for cand in ["cost", "costs", "expenses", "charges_usd", "price", "amount"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "charges"})
                break
    if "charges" not in df.columns:
        raise ValueError(f"No se encontró columna 'charges' en insurance.csv. Columnas: {list(df.columns)}")

    y = df["charges"]
    X = df.drop(columns=["charges"])

    # Columnas esperadas (si alguna falta no pasa nada, solo se ignora)
    cat = [c for c in ["sex", "smoker", "region"] if c in X.columns]
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), cat),
        ("sc", StandardScaler(), num),
    ])

    Xp = pre.fit_transform(X)
    model = LinearRegression().fit(Xp, y)

    joblib.dump(model, MODELS_DIR / "insurance_model.pkl")
    joblib.dump(pre,   MODELS_DIR / "insurance_pipeline.pkl")

# ---------- DIABETES (clasificación) ----------

def _normalize_diabetes_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta y renombra la columna objetivo de diabetes a 'Outcome'.
    Acepta varios alias (case-insensitive) y, si no encuentra,
    intenta elegir una columna binaria 0/1.
    """
    candidates = {
        "outcome", "diabetes", "diabetes_binary", "class", "target",
        "label", "outcome (1=yes; 0=no)", "diabetic"
    }
    rename_map = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in candidates:
            rename_map[c] = "Outcome"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Outcome" not in df.columns:
        # Busca alguna columna estrictamente binaria 0/1
        binary_cols = [c for c in df.columns
                       if df[c].dropna().isin([0, 1]).all() and df[c].nunique(dropna=True) == 2]
        if binary_cols:
            df = df.rename(columns={binary_cols[0]: "Outcome"})

    if "Outcome" not in df.columns:
        raise ValueError(
            f"No se encontró la columna objetivo para diabetes. "
            f"Columnas disponibles: {list(df.columns)}.\n"
            "Renombra en el CSV la columna del diagnóstico a 'Outcome' (0/1), "
            "o agrega su alias en _normalize_diabetes_target()."
        )
    return df

def _clean_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """
    En Pima hay ceros imposibles: BP, SkinThickness, Insulin, BMI -> pásalos a NaN y luego imputa mediana.
    Solo actúa sobre columnas presentes.
    """
    cols_zero_nan = ["BloodPressure", "SkinThickness", "Insulin", "BMI"]
    present = [c for c in cols_zero_nan if c in df.columns]
    for c in present:
        df.loc[df[c] == 0, c] = np.nan
    if present:
        df[present] = df[present].fillna(df[present].median())
    return df

def train_diabetes(df: pd.DataFrame):
    df = _normalize_diabetes_target(df.copy())
    df = _clean_diabetes(df)

    y = df["Outcome"]
    X = df.drop(columns=["Outcome"])

    pre = ColumnTransformer([
        ("sc", StandardScaler(), X.columns)
    ], remainder="drop")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    Xtr_p = pre.fit_transform(Xtr)
    Xte_p = pre.transform(Xte)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr_p, ytr)

    # Umbral óptimo por Youden J
    prob = clf.predict_proba(Xte_p)[:, 1]
    fpr, tpr, thr = roc_curve(yte, prob)
    t_star = float(thr[(tpr - fpr).argmax()])

    joblib.dump(clf, MODELS_DIR / "diabetes_model.pkl")
    joblib.dump(pre,  MODELS_DIR / "diabetes_pipeline.pkl")
    json.dump({"diabetes_threshold": t_star}, open(MODELS_DIR / "threshold.json", "w", encoding="utf-8"))

# ---------- carga de CSV desde /models (como en tu estructura) ----------

def load_csvs_from_models():
    """
    En tu proyecto los CSV están en /models (según la captura).
    """
    ins = _read_csv_auto(MODELS_DIR / "insurance.csv")
    dia = _read_csv_auto(MODELS_DIR / "diabetes.csv")
    return ins, dia
