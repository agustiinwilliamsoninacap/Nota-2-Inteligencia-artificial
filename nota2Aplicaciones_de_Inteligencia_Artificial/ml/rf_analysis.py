# ml/rf_analysis.py
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Reutilizamos utilidades del training para que el target se normalice igual
from .training import (
    load_csvs_from_models,
    _normalize_diabetes_target,  # ya existe en training.py
    _clean_diabetes,             # idem
    MODELS_DIR
)

def rf_insurance_report(df: pd.DataFrame):
    # y: charges (o ya renombrado en training)
    if "charges" not in df.columns:
        raise ValueError(f"No se encontró 'charges' en insurance.csv. Columnas: {list(df.columns)}")
    y = df["charges"]
    X = df.drop(columns=["charges"])

    # One-hot para categóricas (sex, smoker, region si existen)
    Xd = pd.get_dummies(X, drop_first=False)
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=400, random_state=42)
    rf.fit(Xtr, ytr)

    imp = pd.Series(rf.feature_importances_, index=Xtr.columns)\
            .sort_values(ascending=False)
    imp.rename_axis("feature").reset_index(name="importance")\
       .to_csv(MODELS_DIR / "insurance_importances.csv", index=False)

def rf_diabetes_report(df: pd.DataFrame):
    # Normaliza nombre del target y limpia ceros imposibles igual que en training
    df = _normalize_diabetes_target(df.copy())
    df = _clean_diabetes(df)

    y = df["Outcome"]
    X = df.drop(columns=["Outcome"])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=600, random_state=42, class_weight="balanced"
    )
    rf.fit(Xtr, ytr)

    imp = pd.Series(rf.feature_importances_, index=Xtr.columns)\
            .sort_values(ascending=False)
    imp.rename_axis("feature").reset_index(name="importance")\
       .to_csv(MODELS_DIR / "diabetes_importances.csv", index=False)

def run_both_reports():
    ins, dia = load_csvs_from_models()
    rf_insurance_report(ins)
    rf_diabetes_report(dia)
