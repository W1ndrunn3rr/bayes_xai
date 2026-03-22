import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from sklearn.model_selection import StratifiedShuffleSplit

from src.esitmator.bayes_estimator import BayessianClassifier
from src.models.patient_record import PatientRecord
from src.preprocessing.preprocessor import Preprocessor


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    uncertainty: dict[str, float]


preprocessor = Preprocessor()
clf = BayessianClassifier(n_samples=2000, num_warmup=500, num_chains=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = pd.read_csv("data/NSH_clear.csv")
    X = data.drop(columns=["cardio_risk"])
    y = data["cardio_risk"]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, _ in split.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

    X_train_processed = preprocessor.fit_transform(X_train)
    clf.fit(X_train_processed, y_train)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord):
    X = preprocessor.transform([record])

    prediction = int(clf.predict(X)[0])
    probability = float(clf.predict_proba(X)[0, 1])
    uncertainty = clf.predict_uncertainty(X)

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        uncertainty={
            "mean": float(uncertainty["mean"][0]),
            "std": float(uncertainty["std"][0]),
            "lower": float(uncertainty["lower"][0]),
            "upper": float(uncertainty["upper"][0]),
        },
    )


_frontend = Path(__file__).resolve().parents[2] / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("src.server.server:app", host="0.0.0.0", port=8000, reload=False)
