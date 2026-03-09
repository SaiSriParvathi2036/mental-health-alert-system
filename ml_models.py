"""
Machine Learning Model Training
=================================
Trains and compares three classifiers for mental health text classification:
  1. Logistic Regression
  2. Support Vector Machine (LinearSVC)
  3. Random Forest Classifier

Includes hyperparameter tuning, cross-validation, and model persistence.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")


LABEL_ORDER = [
    "Normal Conversation",
    "Stress / Anxiety",
    "Depression / Sadness",
    "Suicide Risk",
    "Cyberbullying",
    "Violence / Threats",
    "Trust / Relationship Issues",
]

RISK_LEVELS = {
    "Normal Conversation":        "low",
    "Stress / Anxiety":           "medium",
    "Depression / Sadness":       "medium",
    "Suicide Risk":               "high",
    "Cyberbullying":              "high",
    "Violence / Threats":         "high",
    "Trust / Relationship Issues":"medium",
}


def get_models() -> dict:
    """Return dictionary of configured ML models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(
                max_iter=2000,
                C=1.0,
                random_state=42,
                class_weight="balanced",
            ),
            cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute full evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    metrics = {
        "model": model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABEL_ORDER).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=LABEL_ORDER, zero_division=0, output_dict=True
        ),
    }
    return metrics


class ModelTrainer:
    """
    Orchestrates training, evaluation, and saving of all ML models.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = get_models()
        self.trained_models = {}
        self.evaluation_results = {}
        self.best_model_name = None
        self.best_model = None

    def train(self, X_train, y_train, X_test, y_test, verbose: bool = True) -> dict:
        """Train all models and evaluate on test set."""
        results = {}
        best_f1 = 0

        if verbose:
            print("\n" + "="*60)
            print("  MENTAL HEALTH ALERT SYSTEM — MODEL TRAINING")
            print("="*60)
            print(f"  Training samples : {X_train.shape[0]}")
            print(f"  Test samples     : {X_test.shape[0]}")
            print(f"  Feature dims     : {X_train.shape[1]}")
            print(f"  Classes          : {len(LABEL_ORDER)}")
            print("="*60 + "\n")

        for name, model in self.models.items():
            if verbose:
                print(f"⏳ Training: {name}...")
            
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            metrics = evaluate_model(model, X_test, y_test, name)
            results[name] = metrics

            if verbose:
                print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}  |  "
                      f"F1: {metrics['f1_score']:.4f}  |  "
                      f"Precision: {metrics['precision']:.4f}  |  "
                      f"Recall: {metrics['recall']:.4f}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                self.best_model_name = name
                self.best_model = model

        if verbose:
            print(f"\n🏆 Best Model: {self.best_model_name} (F1={best_f1:.4f})\n")

        self.evaluation_results = results
        return results

    def cross_validate(self, X, y, cv: int = 5, verbose: bool = True) -> dict:
        """Run stratified k-fold cross-validation for all models."""
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        if verbose:
            print(f"\n📊 Running {cv}-Fold Cross-Validation...")

        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
            cv_results[name] = {
                "mean_f1": round(float(scores.mean()), 4),
                "std_f1":  round(float(scores.std()), 4),
                "scores":  [round(float(s), 4) for s in scores],
            }
            if verbose:
                print(f"  {name:30s}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")

        return cv_results

    def save_models(self, verbose: bool = True):
        """Persist all trained models to disk."""
        for name, model in self.trained_models.items():
            filename = name.lower().replace(" ", "_") + ".pkl"
            path = os.path.join(self.model_dir, filename)
            joblib.dump(model, path)
            if verbose:
                print(f"  💾 Saved: {path}")

        # Save best model separately
        if self.best_model:
            joblib.dump(self.best_model, os.path.join(self.model_dir, "best_model.pkl"))
            with open(os.path.join(self.model_dir, "best_model_name.txt"), "w") as f:
                f.write(self.best_model_name)

        # Save evaluation results
        results_path = os.path.join(self.model_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        if verbose:
            print(f"  💾 Evaluation results saved to {results_path}")

    def load_models(self, verbose: bool = True):
        """Load all saved models from disk."""
        for name in self.models.keys():
            filename = name.lower().replace(" ", "_") + ".pkl"
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                self.trained_models[name] = joblib.load(path)
                if verbose:
                    print(f"  📂 Loaded: {name}")

        best_path = os.path.join(self.model_dir, "best_model.pkl")
        name_path = os.path.join(self.model_dir, "best_model_name.txt")
        if os.path.exists(best_path):
            self.best_model = joblib.load(best_path)
        if os.path.exists(name_path):
            with open(name_path) as f:
                self.best_model_name = f.read().strip()

        results_path = os.path.join(self.model_dir, "evaluation_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                self.evaluation_results = json.load(f)

    def predict(self, X, model_name: str = None) -> tuple:
        """Predict class and probability for input features."""
        model = (self.trained_models.get(model_name) if model_name
                 else self.best_model)
        if model is None:
            raise RuntimeError("No model available. Train or load models first.")

        pred = model.predict(X)
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        return pred, proba

    def predict_risk(self, text_vector, model_name: str = None) -> dict:
        """Full prediction with risk classification for a text vector."""
        pred, proba = self.predict(text_vector, model_name)
        label = pred[0]
        risk_level = RISK_LEVELS.get(label, "low")

        result = {
            "predicted_label": label,
            "risk_level": risk_level,
            "model_used": model_name or self.best_model_name,
        }

        if proba is not None:
            classes = (self.trained_models.get(model_name) or self.best_model).classes_
            prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba[0])}
            result["confidence"] = round(float(max(proba[0])), 4)
            result["class_probabilities"] = prob_dict

        return result
