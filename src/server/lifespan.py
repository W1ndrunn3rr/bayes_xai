import mlflow
import mlflow.sklearn
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = pd.read_csv("data/NSH_clear.csv")
    X = data.drop(columns=["cardio_risk"])
    y = data["cardio_risk"]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preprocessor = app.state.preprocessor
    clf = app.state.clf

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    mlflow.set_experiment("bayes_xai")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "n_samples": clf.n_samples_,
                "num_warmup": clf.num_warmup_,
                "num_chains": clf.num_chains_,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )

        clf.fit(X_train_processed, y_train)

        y_pred = clf.predict(X_test_processed)
        y_proba = clf.predict_proba(X_test_processed)[:, 1]

        uncertainty = clf.predict_uncertainty(X_test_processed)

        mlflow.log_metrics(
            {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_roc_auc": roc_auc_score(y_test, y_proba),
                "uncertainty_mean": uncertainty["mean"].mean(),
                "uncertainty_std": uncertainty["std"].mean(),
                "uncertainty_lower": uncertainty["lower"].mean(),
                "uncertainty_upper": uncertainty["upper"].mean(),
                "uncertainty_mean_std": uncertainty["mean"].std(),
                "uncertainty_CI_width": (
                    uncertainty["upper"] - uncertainty["lower"]
                ).mean(),
            }
        )

        mlflow.sklearn.log_model(clf, artifact_path="model")

    yield
