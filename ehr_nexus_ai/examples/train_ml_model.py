"""
EHR Nexus AI - ML Training Example
====================================
Complete example showing how to train ML models on your patient data.

Usage:
    python -m ehr_nexus_ai.examples.train_ml_model

Steps:
    1. Preview your dataset to understand its structure
    2. Train a symptom correlation model
    3. Train a diagnosis prediction model
    4. Evaluate and compare against rule-based baseline
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ehr_nexus_ai.ml import ModelTrainer, MLConfig
from ehr_nexus_ai.core.orchestrator import DiagnosticCorrelationOrchestrator


def main():
    """Run the ML training example workflow."""

    # ================================================================
    # STEP 0: Setup
    # ================================================================
    print("=" * 60)
    print("EHR Nexus AI - ML Training Example")
    print("=" * 60)

    # Use small dataset config (simpler model, faster training)
    config = MLConfig.small_dataset()
    trainer = ModelTrainer(config)

    # ================================================================
    # STEP 1: Generate a synthetic training dataset for demo purposes.
    #         ** REPLACE THIS with your real CSV file path **
    # ================================================================
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    demo_csv = data_dir / "demo_symptom_pairs.csv"

    if not demo_csv.exists():
        print("\nCreating demo symptom pairs dataset...")
        _create_demo_dataset(demo_csv)

    # ================================================================
    # STEP 2: Preview the dataset
    # ================================================================
    print(f"\n{'-'*60}")
    print("STEP 2: Preview Dataset")
    print(f"{'-'*60}")
    trainer.preview(str(demo_csv))

    # ================================================================
    # STEP 3: Train the model
    # ================================================================
    print(f"\n{'-'*60}")
    print("STEP 3: Train ML Model")
    print(f"{'-'*60}")
    results = trainer.train(
        dataset_path=str(demo_csv),
        task="symptom_correlation",
        save_model=True,
        model_name="demo_symptom_model",
    )

    # ================================================================
    # STEP 4: View training results
    # ================================================================
    print(f"\n{'-'*60}")
    print("STEP 4: Training Results Summary")
    print(f"{'-'*60}")
    print(f"  Accuracy:  {results['test_metrics'].get('accuracy', 0):.2%}")
    print(f"  F1 Score:  {results['test_metrics'].get('f1_score', 0):.4f}")
    print(f"  Grade:     {results.get('grade', 'N/A')}")
    print(f"  Saved to:  {results.get('saved_path', 'N/A')}")

    if results.get("feature_importance_top"):
        print(f"\n  Top features:")
        for i, feat in enumerate(results["feature_importance_top"][:5], 1):
            print(f"    {i}. {feat['feature']} (importance: {feat['importance']:.4f})")

    # ================================================================
    # STEP 5: Use trained model in the orchestrator
    # ================================================================
    print(f"\n{'-'*60}")
    print("STEP 5: Using ML in the Orchestrator")
    print(f"{'-'*60}")

    # Create orchestrator with ML enabled and loaded model
    saved_path = results.get("saved_path")
    if saved_path and os.path.exists(saved_path):
        orchestrator = DiagnosticCorrelationOrchestrator(
            use_ml=True,
            ml_model_path=saved_path,
        )
        print("ML-powered orchestrator initialized.")
        pipeline_info = orchestrator.get_pipeline_description()
        print(f"  ML Enabled:     {pipeline_info['ml_enabled']}")
        print(f"  Model:          {pipeline_info['ml_info'].get('model_name', 'N/A')}")
        print(f"  Model Metrics:  {pipeline_info['ml_info'].get('metrics', {})}")

    # ================================================================
    # STEP 6: Training summary
    # ================================================================
    print(f"\n{'-'*60}")
    print("STEP 6: Full Training History")
    print(f"{'-'*60}")
    print(trainer.training_summary())

    # ================================================================
    # HOW TO USE WITH YOUR REAL DATA
    # ================================================================
    print(f"\n{'-'*60}")
    print("HOW TO USE WITH YOUR OWN CSV DATASET")
    print(f"{'-'*60}")
    print("""
    # 1. Place your CSV file in: ehr_nexus_ai/data/your_dataset.csv
    #
    # 2. Preview it:
    #    trainer = ModelTrainer()
    #    trainer.preview("ehr_nexus_ai/data/your_dataset.csv")
    #
    # 3. Train:
    #    results = trainer.train(
    #        dataset_path="ehr_nexus_ai/data/your_dataset.csv",
    #        task="auto",  # or "symptom_correlation", "diagnosis_prediction", "lab_anomaly"
    #    )
    #
    # 4. Check accuracy:
    #    print(f"Accuracy: {results['test_metrics']['accuracy']:.2%}")
    #
    # 5. Integrate:
    #    orch = DiagnosticCorrelationOrchestrator(use_ml=True)
    #    orch.train_ml("ehr_nexus_ai/data/your_dataset.csv")
    #
    # See dataset_schema.md for required column formats.
    """)

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


def _create_demo_dataset(path: Path):
    """Create a small demo symptom pairs dataset for testing."""
    import csv

    # Symptom pairs: 1 = same/correlated, 0 = different
    pairs = [
        # (symptom_a, symptom_b, label, severity_a, severity_b)
        # Same symptoms (synonyms)
        ("headache", "head pain", 1, "mild", "moderate"),
        ("headache", "cephalgia", 1, "mild", "severe"),
        ("fever", "pyrexia", 1, "high", "high"),
        ("fever", "elevated temperature", 1, "high", "moderate"),
        ("cough", "coughing", 1, "moderate", "moderate"),
        ("nausea", "feeling sick", 1, "mild", "mild"),
        ("nausea", "queasy", 1, "moderate", "mild"),
        ("vomiting", "throwing up", 1, "severe", "severe"),
        ("dizziness", "vertigo", 1, "moderate", "severe"),
        ("dizziness", "lightheaded", 1, "mild", "mild"),
        ("fatigue", "tiredness", 1, "moderate", "moderate"),
        ("fatigue", "exhaustion", 1, "severe", "severe"),
        ("chest pain", "chest discomfort", 1, "severe", "moderate"),
        ("chest pain", "angina", 1, "severe", "severe"),
        ("shortness of breath", "dyspnea", 1, "severe", "severe"),
        ("shortness of breath", "difficulty breathing", 1, "severe", "moderate"),
        ("abdominal pain", "stomach ache", 1, "moderate", "moderate"),
        ("abdominal pain", "belly pain", 1, "moderate", "mild"),
        ("sore throat", "throat pain", 1, "mild", "mild"),
        ("sore throat", "scratchy throat", 1, "mild", "mild"),
        ("rash", "skin eruption", 1, "moderate", "moderate"),
        ("joint pain", "arthralgia", 1, "moderate", "severe"),
        ("joint pain", "joint ache", 1, "moderate", "moderate"),
        ("swelling", "edema", 1, "moderate", "moderate"),
        ("back pain", "backache", 1, "severe", "severe"),
        ("back pain", "lumbago", 1, "severe", "moderate"),

        # Different symptoms (negative examples)
        ("headache", "fever", 0, "mild", "high"),
        ("headache", "cough", 0, "moderate", "moderate"),
        ("headache", "nausea", 0, "severe", "mild"),
        ("fever", "cough", 0, "high", "moderate"),
        ("fever", "dizziness", 0, "high", "mild"),
        ("cough", "nausea", 0, "moderate", "mild"),
        ("cough", "chest pain", 0, "moderate", "severe"),
        ("nausea", "dizziness", 0, "moderate", "moderate"),
        ("nausea", "fatigue", 0, "mild", "severe"),
        ("vomiting", "headache", 0, "severe", "mild"),
        ("dizziness", "chest pain", 0, "moderate", "severe"),
        ("fatigue", "fever", 0, "severe", "high"),
        ("chest pain", "abdominal pain", 0, "severe", "moderate"),
        ("shortness of breath", "fatigue", 0, "severe", "moderate"),
        ("abdominal pain", "back pain", 0, "moderate", "severe"),
        ("sore throat", "joint pain", 0, "mild", "moderate"),
        ("rash", "fever", 0, "moderate", "high"),
        ("back pain", "nausea", 0, "severe", "mild"),
        ("joint pain", "headache", 0, "moderate", "mild"),
        ("swelling", "cough", 0, "mild", "moderate"),
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symptom_a", "symptom_b", "label", "severity_a", "severity_b"])
        for pair in pairs:
            writer.writerow(pair)

    print(f"Created demo dataset: {path} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()