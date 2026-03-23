from src.esitmator.bayes_estimator import BayessianClassifier
from src.preprocessing.preprocessor import Preprocessor
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    data = pd.read_csv("data/NSH_clear.csv")

    X = data.drop(columns=["cardio_risk"])
    y = data["cardio_risk"]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    clf = BayessianClassifier(n_samples=2000, num_warmup=500, num_chains=1)
    clf.fit(X_train_processed, y_train)

    y_pred = clf.predict(X_test_processed)
    y_proba = clf.predict_proba(X_test_processed)[:, 1]
    uncertainty = clf.predict_uncertainty(X_test_processed)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=axs[0])
    axs[0].set_title("Confusion matrix")

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axs[1])
    axs[1].set_title("ROC Curve")

    axs[2].hist(uncertainty["std"], bins=30, color="#534AB7", edgecolor="white")
    axs[2].axvline(
        uncertainty["std"].mean(),
        color="red",
        linestyle="--",
        label=f"mean std={uncertainty['std'].mean():.2f}",
    )
    axs[2].set_title("Model uncertainty distribution")
    axs[2].set_xlabel("prediction std")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
