"""
Prepare external medical datasets for EHR Nexus ML training.

This script reads selected external datasets, normalizes their
column names to the schema expected by ModelTrainer, and writes training-ready
CSV files into ehr_nexus_ai/data/.

Usage:
    python -m ehr_nexus_ai.examples.prepare_training_data "D:\\(Datasets)"
"""

import sys
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    unnamed = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _write(df: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows, {len(df.columns)} columns)")
    return path


def prepare_disease_prediction_from_training_testing(dataset_root: Path) -> Path:
    """Convert symptom-indicator disease files to diagnosis_prediction schema."""
    base = dataset_root / "New Zip"
    frames = []
    for name in ("Training.csv", "Testing.csv"):
        df = pd.read_csv(base / name)
        df = _clean_columns(df)
        df = df.rename(columns={"prognosis": "confirmed_diagnosis"})
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    return _write(combined, "diagnosis_training_symptom_indicators.csv")


def prepare_healthcare_symptoms(dataset_root: Path) -> Path:
    """Convert text symptom list dataset to diagnosis_prediction schema."""
    src = dataset_root / "New Zip" / "Healthcare.csv"
    df = _clean_columns(pd.read_csv(src))
    df = df.rename(
        columns={
            "Patient_ID": "patient_id",
            "Age": "age",
            "Gender": "sex",
            "Symptoms": "symptom_names",
            "Disease": "confirmed_diagnosis",
        }
    )
    symptoms = (
        df["symptom_names"]
        .fillna("")
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.split(",")
    )
    all_symptoms = sorted(
        {
            symptom.strip().replace(" ", "_")
            for row in symptoms
            for symptom in row
            if symptom.strip()
        }
    )
    for symptom in all_symptoms:
        df[f"symptom_{symptom}"] = symptoms.apply(
            lambda row, value=symptom: int(
                value in {item.strip().replace(" ", "_") for item in row}
            )
        )

    keep = ["patient_id", "age", "sex", "confirmed_diagnosis"] + [
        f"symptom_{symptom}" for symptom in all_symptoms
    ]
    return _write(df[keep].drop_duplicates(), "diagnosis_training_healthcare.csv")


def prepare_synthetic_medical_symptoms(dataset_root: Path) -> Path:
    """Convert synthetic symptom/lab dataset to diagnosis_prediction schema."""
    src = dataset_root / "New Zip" / "synthetic_medical_symptoms_dataset.csv"
    df = _clean_columns(pd.read_csv(src))
    df = df.rename(
        columns={
            "gender": "sex",
            "diagnosis": "confirmed_diagnosis",
        }
    )
    return _write(df.drop_duplicates(), "diagnosis_training_synthetic.csv")


def prepare_drug_recommendation(dataset_root: Path) -> Path:
    """Normalize drug prescription data for future medication-order modeling."""
    src = dataset_root / "New Zip" / "Drug prescription Dataset.csv"
    df = _clean_columns(pd.read_csv(src))
    df = df.rename(
        columns={
            "disease": "confirmed_diagnosis",
            "drug": "recommended_drug",
        }
    )
    return _write(df.drop_duplicates(), "medication_recommendation_training.csv")


def main() -> None:
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets")
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. "
            "Pass the dataset folder path as the first argument."
        )

    prepare_disease_prediction_from_training_testing(dataset_root)
    prepare_healthcare_symptoms(dataset_root)
    prepare_synthetic_medical_symptoms(dataset_root)
    prepare_drug_recommendation(dataset_root)


if __name__ == "__main__":
    main()
