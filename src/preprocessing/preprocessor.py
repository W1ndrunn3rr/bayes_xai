import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ..models.patient_record import PatientRecord


class Preprocessor(BaseEstimator, TransformerMixin):
    CAT_COLS = ["sickle_cell_genotype"]
    NUM_COLS = [
        "age",
        "sex",
        "cigs_per_day",
        "malaria_exposure",
        "hemoglobin_g_per_dL",
        "heart_rate_bpm",
        "cholesterol_mg_per_dL",
        "blood_pressure_upper",
        "blood_pressure_lower",
    ]

    def __init__(self) -> None:
        self._pipeline = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.NUM_COLS,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.CAT_COLS,
                ),
            ]
        )

    def _to_dataframe(
        self, records: list[PatientRecord] | pd.DataFrame
    ) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records
        return pd.DataFrame([r.model_dump() for r in records])

    def fit(
        self, records: list[PatientRecord] | pd.DataFrame, y=None
    ) -> "Preprocessor":
        self._pipeline.fit(self._to_dataframe(records))
        return self

    def transform(
        self, records: list[PatientRecord] | pd.DataFrame, y=None
    ) -> np.ndarray:
        return self._pipeline.transform(self._to_dataframe(records))
