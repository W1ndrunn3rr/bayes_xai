from fastapi import APIRouter, Request

from src.models.patient_record import PatientRecord
from src.server.schemas import PredictionResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord, request: Request):
    preprocessor = request.app.state.preprocessor
    clf = request.app.state.clf

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
