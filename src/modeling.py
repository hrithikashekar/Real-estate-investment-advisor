"""Portable model saving/loading.

Pickled XGBoost boosters can fail to load on a different OS/build ("input stream corrupted").
We therefore store the sklearn preprocessor with joblib and the XGBoost model in XGBoost's own
JSON format, and rebuild a small pipeline-like object at load time.
"""
from pathlib import Path
import joblib
from xgboost import XGBClassifier, XGBRegressor


class NativePipeline:
    def __init__(self, prep, model):
        self.prep, self.model = prep, model

    def predict(self, X):
        return self.model.predict(self.prep.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.prep.transform(X))


def save_portable(pipe, models_dir: Path, name: str):
    """name: 'classifier' or 'regressor'. XGBoost -> JSON + preprocessor; otherwise a plain joblib pipeline."""
    model = pipe.named_steps["model"]
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        joblib.dump(pipe.named_steps["prep"], models_dir / f"{name}_prep.joblib", compress=3)
        model.save_model(str(models_dir / f"{name}_xgb.json"))
        (models_dir / f"{name}.joblib").unlink(missing_ok=True)
    else:
        joblib.dump(pipe, models_dir / f"{name}.joblib", compress=3)
        for f in (f"{name}_prep.joblib", f"{name}_xgb.json"):
            (models_dir / f).unlink(missing_ok=True)


def load_portable(models_dir: Path, name: str):
    js = models_dir / f"{name}_xgb.json"
    if js.exists():
        model = XGBClassifier() if name == "classifier" else XGBRegressor()
        model.load_model(str(js))
        return NativePipeline(joblib.load(models_dir / f"{name}_prep.joblib"), model)
    return joblib.load(models_dir / f"{name}.joblib")


def models_exist(models_dir: Path) -> bool:
    return all((models_dir / f"{n}_xgb.json").exists() or (models_dir / f"{n}.joblib").exists()
               for n in ("classifier", "regressor"))
