"""
Ensemble Trainer - Train multiple models and combine predictions.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger('trading_bot')


class EnsembleTrainer:
    """Train ensemble of models for high-confidence predictions."""

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names: List[str] = []
        self.training_metrics: Dict[str, Any] = {}

    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        use_feature_selection: bool = True,
        top_features: int = 60
    ) -> Dict[str, float]:
        """
        Train ensemble of 3 models.

        Models:
        1. XGBoost
        2. Random Forest
        3. Gradient Boosting

        Only trade when ALL models agree with high confidence.
        """
        from xgboost import XGBClassifier

        self.feature_names = list(feature_names)

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Feature Selection
        if use_feature_selection and len(feature_names) > top_features:
            print(f"\nFeature Selection: Selecting top {top_features} features...")
            self.feature_selector = SelectKBest(f_classif, k=top_features)
            X_train = self.feature_selector.fit_transform(X_train, y_train)
            X_test = self.feature_selector.transform(X_test)

            selected_mask = self.feature_selector.get_support()
            self.feature_names = [f for f, m in zip(feature_names, selected_mask) if m]
            print(f"Selected features: {len(self.feature_names)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Model 1: XGBoost
        print("\n[1/3] Training XGBoost...")
        self.models['xgboost'] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.models['xgboost'].fit(X_train_scaled, y_train)

        # Train Model 2: Random Forest
        print("[2/3] Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)

        # Train Model 3: Gradient Boosting
        print("[3/3] Training Gradient Boosting...")
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gradient_boosting'].fit(X_train_scaled, y_train)

        # Evaluate individual models
        print("\n" + "=" * 60)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("=" * 60)

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name:20}: Accuracy = {acc:.4f}")

        # Evaluate ensemble
        print("\n" + "=" * 60)
        print("ENSEMBLE PERFORMANCE")
        print("=" * 60)

        metrics = self._evaluate_ensemble(X_test_scaled, y_test)
        self.training_metrics = metrics

        return metrics

    def _evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble with voting and high-confidence filtering."""
        # Get predictions from all models
        preds = {}
        probas = {}

        for name, model in self.models.items():
            preds[name] = model.predict(X_test)
            probas[name] = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (BUY)

        # Majority voting
        all_preds = np.vstack([preds[name] for name in self.models.keys()]).T
        majority_vote = (all_preds.sum(axis=1) >= 2).astype(int)  # 2 out of 3 agree

        # Unanimous agreement
        unanimous = (all_preds.sum(axis=1) == 0) | (all_preds.sum(axis=1) == 3)
        y_unanimous = np.where(all_preds.sum(axis=1) == 3, 1, 0)

        # Average probability
        avg_proba = np.mean([probas[name] for name in self.models.keys()], axis=0)

        # High confidence (avg proba > 0.65 or < 0.35)
        high_conf_mask = (avg_proba > 0.65) | (avg_proba < 0.35)

        # SUPER HIGH confidence (avg proba > 0.75 or < 0.25)
        super_high_conf_mask = (avg_proba > 0.75) | (avg_proba < 0.25)

        # Calculate metrics
        print("\n--- Majority Voting (2/3) ---")
        acc_majority = accuracy_score(y_test, majority_vote)
        print(f"Accuracy: {acc_majority:.4f}")

        print("\n--- Unanimous Agreement (3/3) ---")
        if unanimous.sum() > 0:
            y_test_unan = y_test[unanimous]
            y_pred_unan = y_unanimous[unanimous]
            acc_unan = accuracy_score(y_test_unan, y_pred_unan)
            print(f"Samples with agreement: {unanimous.sum()} ({unanimous.sum()/len(y_test)*100:.1f}%)")
            print(f"Accuracy: {acc_unan:.4f}")
        else:
            acc_unan = 0
            print("No unanimous predictions")

        print("\n--- High Confidence (prob > 0.65 or < 0.35) ---")
        if high_conf_mask.sum() > 0:
            y_test_hc = y_test[high_conf_mask]
            y_pred_hc = majority_vote[high_conf_mask]
            acc_hc = accuracy_score(y_test_hc, y_pred_hc)
            print(f"High confidence samples: {high_conf_mask.sum()} ({high_conf_mask.sum()/len(y_test)*100:.1f}%)")
            print(f"Accuracy: {acc_hc:.4f}")
        else:
            acc_hc = 0

        print("\n--- SUPER High Confidence (prob > 0.75 or < 0.25) ---")
        if super_high_conf_mask.sum() > 0:
            y_test_shc = y_test[super_high_conf_mask]
            y_pred_shc = majority_vote[super_high_conf_mask]
            acc_shc = accuracy_score(y_test_shc, y_pred_shc)
            print(f"Super high confidence samples: {super_high_conf_mask.sum()} ({super_high_conf_mask.sum()/len(y_test)*100:.1f}%)")
            print(f"Accuracy: {acc_shc:.4f}")
        else:
            acc_shc = 0

        # Unanimous + High Confidence (BEST FILTER)
        best_mask = unanimous & high_conf_mask
        print("\n--- BEST: Unanimous + High Confidence ---")
        if best_mask.sum() > 0:
            y_test_best = y_test[best_mask]
            y_pred_best = y_unanimous[best_mask]
            acc_best = accuracy_score(y_test_best, y_pred_best)
            prec_best = precision_score(y_test_best, y_pred_best, zero_division=0)
            rec_best = recall_score(y_test_best, y_pred_best, zero_division=0)
            f1_best = f1_score(y_test_best, y_pred_best, zero_division=0)

            print(f"Qualifying samples: {best_mask.sum()} ({best_mask.sum()/len(y_test)*100:.1f}%)")
            print(f"Accuracy:  {acc_best:.4f}")
            print(f"Precision: {prec_best:.4f}")
            print(f"Recall:    {rec_best:.4f}")
            print(f"F1 Score:  {f1_best:.4f}")
        else:
            acc_best = 0
            prec_best = rec_best = f1_best = 0
            print("No samples meet criteria")

        return {
            'accuracy': acc_majority,
            'accuracy_unanimous': acc_unan,
            'accuracy_high_conf': acc_hc,
            'accuracy_super_high_conf': acc_shc,
            'accuracy_best': acc_best,
            'precision': prec_best,
            'recall': rec_best,
            'f1': f1_best,
            'unanimous_count': int(unanimous.sum()),
            'high_conf_count': int(high_conf_mask.sum()),
            'best_count': int(best_mask.sum())
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.

        Returns:
            Tuple of (majority_vote, avg_probability, unanimous_mask)
        """
        # Apply feature selection if used
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)

        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        preds = {}
        probas = {}

        for name, model in self.models.items():
            preds[name] = model.predict(X_scaled)
            probas[name] = model.predict_proba(X_scaled)[:, 1]

        # Majority voting
        all_preds = np.vstack([preds[name] for name in self.models.keys()]).T
        majority_vote = (all_preds.sum(axis=1) >= 2).astype(int)

        # Average probability
        avg_proba = np.mean([probas[name] for name in self.models.keys()], axis=0)

        # Unanimous agreement
        unanimous = (all_preds.sum(axis=1) == 0) | (all_preds.sum(axis=1) == 3)

        return majority_vote, avg_proba, unanimous

    def save_model(self, name: str = 'ensemble_model'):
        """Save all models."""
        model_path = os.path.join(self.model_dir, f"{name}.joblib")

        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'metrics': self.training_metrics
        }, model_path)

        print(f"\n✓ Ensemble model saved to: {model_path}")

    def load_model(self, name: str = 'ensemble_model'):
        """Load saved models."""
        model_path = os.path.join(self.model_dir, f"{name}.joblib")

        data = joblib.load(model_path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_selector = data['feature_selector']
        self.feature_names = data['feature_names']
        self.training_metrics = data['metrics']

        print(f"✓ Ensemble model loaded: {name}")
        print(f"  Models: {list(self.models.keys())}")
        print(f"  Features: {len(self.feature_names)}")
