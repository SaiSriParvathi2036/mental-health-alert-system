"""
Train Pipeline - Mental Health Early Alert System
===================================================
Run this file first to train all models.
Command: python train_pipeline.py
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Local imports ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp_preprocessing import NLPPreprocessor
from feature_extraction import TFIDFFeatureExtractor
from ml_models import ModelTrainer, RISK_LEVELS, LABEL_ORDER
from alert_system import AlertSystem
from visualization import (
    plot_category_distribution,
    plot_risk_levels,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_full_dashboard,
)


def banner(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def run_pipeline(
    test_size: float = 0.2,
    output_dir: str = "outputs",
    model_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── STEP 1: Load Dataset ─────────────────────────────────
    banner("STEP 1 — Loading Dataset")

    dataset_path = "mental_health_dataset.csv"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("   Please place mental_health_dataset.csv in the same folder as this file.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded dataset: {len(df)} samples")
    print(f"\nDataset shape : {df.shape}")
    print(f"Columns       : {list(df.columns)}")
    print(f"Classes       : {df['label'].nunique()}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts().to_string())

    # ── STEP 2: NLP Preprocessing ────────────────────────────
    banner("STEP 2 — NLP Preprocessing")

    preprocessor = NLPPreprocessor(remove_stopwords=True, lemmatize=True)

    # Show pipeline demo on one sample
    demo_text = "I've been feeling SO hopeless lately... can't stop crying https://link.com !!!"
    steps = preprocessor.explain_steps(demo_text)
    print("\n  Pipeline demo on sample text:")
    for k, v in steps.items():
        if k == "4_tokens":
            continue
        display = str(v)[:80]
        print(f"  [{k:<20}] {display}")

    print("\n⏳ Preprocessing all texts...")
    df["clean_text"] = preprocessor.preprocess_series(df["text"])
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"✅ Preprocessed {len(df)} valid samples")

    # ── STEP 3: Feature Extraction ───────────────────────────
    banner("STEP 3 — TF-IDF Feature Extraction")

    X_all = df["clean_text"]
    y_all = df["label"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=42,
        stratify=y_all,
    )

    extractor = TFIDFFeatureExtractor(max_features=8000, ngram_range=(1, 2))
    X_train = extractor.fit_transform(X_train_raw)
    X_test = extractor.transform(X_test_raw)
    extractor.save(os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    print(f"\nTraining set  : {X_train.shape}")
    print(f"Test set      : {X_test.shape}")
    print(f"\nTop TF-IDF terms:")
    for term, score in list(extractor.get_top_terms(X_train, n=12).items()):
        print(f"  {term:<25} {score:.4f}")

    # ── STEP 4: Model Training ───────────────────────────────
    banner("STEP 4 — Model Training")

    trainer = ModelTrainer(model_dir=model_dir)
    results = trainer.train(X_train, y_train, X_test, y_test, verbose=True)
    trainer.save_models()

    # ── STEP 5: Detailed Evaluation ──────────────────────────
    banner("STEP 5 — Detailed Evaluation")

    print(f"\n{'Model':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 65)
    for model_name, m in results.items():
        print(
            f"  {model_name:<28} {m['accuracy']:>9.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>8.4f} {m['f1_score']:>8.4f}"
        )
    print("-" * 65)
    print(f"\n🏆 Best model: {trainer.best_model_name}")

    # ── STEP 6: Cross-Validation ─────────────────────────────
    banner("STEP 6 — Cross-Validation (5-Fold)")
    try:
        cv_results = trainer.cross_validate(X_train, y_train, cv=5, verbose=True)
        with open(os.path.join(output_dir, "cv_results.json"), "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"✅ CV results saved")
    except Exception as e:
        print(f"  CV skipped: {e}")

    # ── STEP 7: Visualizations ───────────────────────────────
    banner("STEP 7 — Generating Visualizations")

    print("  Plotting category distribution...")
    p1 = plot_category_distribution(df, output_dir)
    print(f"  ✅ {p1}")

    print("  Plotting risk level breakdown...")
    p2 = plot_risk_levels(df, output_dir)
    print(f"  ✅ {p2}")

    print("  Plotting model comparison...")
    p3 = plot_model_comparison(results, output_dir)
    print(f"  ✅ {p3}")

    print("  Plotting confusion matrices...")
    for name, res in results.items():
        p = plot_confusion_matrix(res["confusion_matrix"], name, LABEL_ORDER, output_dir)
        print(f"  ✅ {p}")

    print("  Plotting per-class F1 scores...")
    p5 = plot_per_class_f1(results, output_dir)
    print(f"  ✅ {p5}")

    print("  Generating full dashboard...")
    p6 = plot_full_dashboard(df, results, output_dir)
    print(f"  ✅ {p6}")

    # ── STEP 8: Demo Predictions + Alerts ───────────────────
    banner("STEP 8 — Live Prediction & Alert Demo")

    alerter = AlertSystem(
        log_dir=os.path.join(output_dir, "alerts"),
        simulation_mode=True,
    )

    test_cases = [
        ("Had a great day with friends at the park today!", "Expected: Normal"),
        ("I can't stop the anxiety, everything feels overwhelming right now.", "Expected: Stress"),
        ("I don't want to be here anymore, I've written my goodbye note.", "Expected: Suicide Risk"),
        ("I'll hurt you if you don't comply, I know where you live.", "Expected: Violence"),
        ("You're worthless, everyone hates you, go away forever.", "Expected: Cyberbullying"),
        ("I don't trust anyone after the betrayal, completely broken.", "Expected: Trust Issues"),
    ]

    print()
    for raw_text, expected in test_cases:
        clean = preprocessor.preprocess(raw_text)
        vec = extractor.transform(pd.Series([clean]))
        pred = trainer.predict_risk(vec)

        label = pred["predicted_label"]
        risk = pred["risk_level"]
        conf = pred.get("confidence", 0)

        icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
        print(f"  {icon} [{risk.upper():<6}] {label:<32} conf={conf:.2f}")

        if len(raw_text) > 60:
            print(f"       Input : \"{raw_text[:60]}...\"")
        else:
            print(f"       Input : \"{raw_text}\"")

        print(f"       {expected}")

        if risk != "low":
            record = alerter.process_alert(raw_text, label, risk, conf)
            print(f"       Alert : {record.alert_id} → {record.channels_notified}")
        print()

    # ── STEP 9: Save Metadata ────────────────────────────────
    banner("STEP 9 — Saving Project Metadata")

    metadata = {
        "dataset_size": len(df),
        "classes": LABEL_ORDER,
        "risk_levels": RISK_LEVELS,
        "best_model": trainer.best_model_name,
        "model_metrics": {
            m: {k: v for k, v in r.items()
                if k not in ["confusion_matrix", "classification_report"]}
            for m, r in results.items()
        },
        "features": extractor.vectorizer.max_features,
        "ngram_range": list(extractor.vectorizer.ngram_range),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Metadata saved to {output_dir}/metadata.json")

    # ── DONE ─────────────────────────────────────────────────
    banner("✅ TRAINING PIPELINE COMPLETE")
    print(f"  Models saved   → {model_dir}/")
    print(f"  Charts saved   → {output_dir}/")
    print(f"  Best model     → {trainer.best_model_name}")
    print(f"\n  ▶ Now run the dashboard:")
    print(f"    streamlit run app_streamlit.py")
    print()

    return {
        "trainer": trainer,
        "extractor": extractor,
        "preprocessor": preprocessor,
        "results": results,
        "df": df,
    }


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()