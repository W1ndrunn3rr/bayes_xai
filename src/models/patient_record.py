from pydantic import BaseModel, field_validator


class PatientRecord(BaseModel):
    age: int
    sex: int
    cigs_per_day: int
    sickle_cell_genotype: str
    malaria_exposure: float
    hemoglobin_g_per_dL: float
    heart_rate_bpm: int
    cholesterol_mg_per_dL: float
    blood_pressure_upper: float
    blood_pressure_lower: float

    @field_validator("sex")
    @classmethod
    def sex_must_be_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("sex must be 0 (female) or 1 (male)")
        return v

    @field_validator("malaria_exposure")
    @classmethod
    def malaria_exposure_valid(cls, v: float) -> float:
        if v not in (0.0, 0.5, 1.0):
            raise ValueError("malaria_exposure must be 0.0, 0.5, or 1.0")
        return v
