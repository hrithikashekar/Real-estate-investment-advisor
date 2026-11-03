"""
Steps 3-4 - Model training, evaluation and MLflow experiment tracking.

Run:  python src/train.py            (then:  mlflow ui --backend-store-uri sqlite:///mlflow.db)
Out:  models/classifier.joblib, models/regressor.joblib, models/metrics.json,
      models/feature_importance_*.csv, models/feature_config.json
"""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from modeling import save_portable

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SEED = 42

NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Floor_No", "Total_Floors",
           "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Parking_Flag", "Security_Flag",
           "Under_Construction", "Transport_Score", "School_Density_Score", "Hospital_Score", "Infra_Score",
           "Floor_Ratio", "Size_per_BHK", "City_Median_PPS", "Locality_Median_PPS", "PPS_vs_City", "Is_Tier1"]
CATEGORICAL = ["City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
               "Facing", "Owner_Type", "Age_Bucket"]
FEATURES = NUMERIC + CATEGORICAL
# targets / label-derived columns are deliberately NOT features (prevents leakage)


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def clf_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=500, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=20,
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED,
                                 eval_metric="logloss"),
    }


def reg_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=20, max_features=0.4,
                                              max_samples=0.3, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED),
    }


def importance(pipe, top=20):
    names = pipe.named_steps["prep"].get_feature_names_out()
    m = pipe.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        vals = m.feature_importances_
    else:
        vals = np.abs(np.ravel(m.coef_))
    s = pd.Series(vals, index=[n.split("__", 1)[1] for n in names]).sort_values(ascending=False)
    return s.head(top)


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("Real_Estate_Investment_Advisor")
    df = pd.read_csv(ROOT / "data" / "cleaned_housing.csv")
    X = df[FEATURES]
    y_clf, y_reg = df["Good_Investment"], df["Future_Price_5Y"]
    Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=SEED, stratify=y_clf)
    print(f"train {len(Xtr):,}  test {len(Xte):,}  positive rate {y_clf.mean():.3f}")

    results = {"classification": {}, "regression": {}}
    best = {}

    # ------------------------- classification -------------------------
    for name, model in clf_models().items():
        with mlflow.start_run(run_name=f"clf_{name}"):
            t0 = time.time()
            pipe = Pipeline([("prep", preprocessor()), ("model", model)]).fit(Xtr, ytr_c)
            proba = pipe.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            m = dict(accuracy=accuracy_score(yte_c, pred), precision=precision_score(yte_c, pred),
                     recall=recall_score(yte_c, pred), f1=f1_score(yte_c, pred),
                     roc_auc=roc_auc_score(yte_c, proba), train_seconds=time.time() - t0)
            mlflow.set_tags({"task": "classification", "target": "Good_Investment"})
            mlflow.log_params({"model": name, **{k: str(v) for k, v in model.get_params().items()
                                                 if k in ("n_estimators", "max_depth", "learning_rate", "C")}})
            mlflow.log_metrics(m)
            results["classification"][name] = {k: round(float(v), 4) for k, v in m.items()}
            results["classification"][name]["confusion_matrix"] = confusion_matrix(yte_c, pred).tolist()
            print("CLF", name, {k: round(float(v), 4) for k, v in m.items()})
            if name not in best.get("clf_pipes", {}):
                best.setdefault("clf_pipes", {})[name] = pipe

    # ------------------------- regression -------------------------
    for name, model in reg_models().items():
        with mlflow.start_run(run_name=f"reg_{name}"):
            t0 = time.time()
            pipe = Pipeline([("prep", preprocessor()), ("model", model)]).fit(Xtr, ytr_r)
            pred = pipe.predict(Xte)
            m = dict(rmse=float(np.sqrt(mean_squared_error(yte_r, pred))), mae=mean_absolute_error(yte_r, pred),
                     r2=r2_score(yte_r, pred), train_seconds=time.time() - t0)
            mlflow.set_tags({"task": "regression", "target": "Future_Price_5Y"})
            mlflow.log_params({"model": name, **{k: str(v) for k, v in model.get_params().items()
                                                 if k in ("n_estimators", "max_depth", "learning_rate")}})
            mlflow.log_metrics(m)
            results["regression"][name] = {k: round(float(v), 4) for k, v in m.items()}
            print("REG", name, {k: round(float(v), 4) for k, v in m.items()})
            best.setdefault("reg_pipes", {})[name] = pipe
            best.setdefault("reg_resid", {})[name] = (yte_r.values - pred)

    # ------------------------- select + save best -------------------------
    best_c = max(results["classification"], key=lambda k: results["classification"][k]["f1"])
    best_r = min(results["regression"], key=lambda k: results["regression"][k]["rmse"])
    MODELS.mkdir(exist_ok=True)
    clf_pipe, reg_pipe = best["clf_pipes"][best_c], best["reg_pipes"][best_r]
    save_portable(clf_pipe, MODELS, "classifier")
    save_portable(reg_pipe, MODELS, "regressor")
    importance(clf_pipe).to_csv(MODELS / "feature_importance_classifier.csv", header=["importance"])
    importance(reg_pipe).to_csv(MODELS / "feature_importance_regressor.csv", header=["importance"])

    resid = best["reg_resid"][best_r]
    results["best"] = {"classifier": best_c, "regressor": best_r,
                       "residual_p10": float(np.percentile(resid, 10)), "residual_p90": float(np.percentile(resid, 90))}
    results["data"] = {"rows": int(len(df)), "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
                       "positive_rate": float(y_clf.mean())}
    (MODELS / "metrics.json").write_text(json.dumps(results, indent=2))
    (MODELS / "feature_config.json").write_text(json.dumps({"numeric": NUMERIC, "categorical": CATEGORICAL}))

    with mlflow.start_run(run_name="final_selection"):
        mlflow.set_tags({"task": "summary"})
        mlflow.log_param("best_classifier", best_c)
        mlflow.log_param("best_regressor", best_r)
        mlflow.log_metrics({"best_clf_f1": results["classification"][best_c]["f1"],
                            "best_reg_rmse": results["regression"][best_r]["rmse"]})
        for f in sorted(MODELS.glob("classifier*")) + sorted(MODELS.glob("regressor*")) + [MODELS / "metrics.json"]:
            mlflow.log_artifact(str(f))
    print("best:", results["best"])


if __name__ == "__main__":
    main()
