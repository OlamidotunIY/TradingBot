"""
ML Trainer - Train and evaluate ML models for trading.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger('trading_bot')


class MLTrainer:
    """Train and evaluate ML models for trading signals."""

    def __init__(self, model_dir: str = 'models'):
        """
        Initialize ML Trainer.

        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None  # For feature selection
        self.feature_names: List[str] = []
        self.all_feature_names: List[str] = []  # All features before selection
        self.training_metrics: Dict[str, Any] = {}

    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        n_estimators: int = 200,
        max_depth: int = 15,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train Random Forest classifier.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            test_size: Fraction for test set
            n_estimators: Number of trees
            max_depth: Max tree depth
            random_state: Random seed

        Returns:
            Dict of evaluation metrics
        """
        self.feature_names = feature_names

        # Time-based split (don't shuffle for time series!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print(f"\nTraining Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        metrics = self._evaluate(X_test_scaled, y_test)
        self.training_metrics = metrics

        # Feature importance
        self._print_feature_importance(top_n=20)

        return metrics

    def train_gradient_boosting(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1
    ) -> Dict[str, float]:
        """Train Gradient Boosting classifier."""
        self.feature_names = feature_names

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTraining Gradient Boosting...")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        metrics = self._evaluate(X_test_scaled, y_test)
        self.training_metrics = metrics

        return metrics

    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_feature_selection: bool = True,
        top_features: int = 50
    ) -> Dict[str, float]:
        """
        Train XGBoost classifier with optional feature selection.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            test_size: Fraction for test set
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth
            learning_rate: Learning rate
            use_feature_selection: If True, select top features first
            top_features: Number of features to keep if feature_selection is True
        """
        from xgboost import XGBClassifier
        from sklearn.feature_selection import SelectKBest, f_classif

        self.feature_names = list(feature_names)

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Feature Selection
        if use_feature_selection and len(feature_names) > top_features:
            print(f"\nFeature Selection: Selecting top {top_features} features from {len(feature_names)}...")
            self.feature_selector = SelectKBest(f_classif, k=top_features)
            X_train = self.feature_selector.fit_transform(X_train, y_train)
            X_test = self.feature_selector.transform(X_test)

            # Update feature names
            selected_mask = self.feature_selector.get_support()
            self.feature_names = [f for f, m in zip(feature_names, selected_mask) if m]
            print(f"Selected features: {len(self.feature_names)}")
        else:
            self.feature_selector = None

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTraining XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        metrics = self._evaluate_binary(X_test_scaled, y_test)
        self.training_metrics = metrics

        # Feature importance
        self._print_feature_importance(top_n=20)

        return metrics

    def _evaluate_binary(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate binary classification model."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0),
            'f1': f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        }

        print("\n" + "=" * 60)
        print("MODEL EVALUATION (BINARY)")
        print("=" * 60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['SELL (0)', 'BUY (1)'],
                                    zero_division=0))

        # Confusion matrix summary
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"  True SELL predicted SELL: {cm[0][0]}")
        print(f"  True SELL predicted BUY:  {cm[0][1]}")
        print(f"  True BUY predicted SELL:  {cm[1][0]}")
        print(f"  True BUY predicted BUY:   {cm[1][1]}")

        return metrics

    def walk_forward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Walk-forward validation for time series.

        More realistic than random split - trains on past, tests on future.
        """
        self.feature_names = feature_names

        tscv = TimeSeriesSplit(n_splits=n_splits)

        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        print(f"\nWalk-Forward Validation ({n_splits} folds)...")
        print("-" * 60)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            all_metrics['accuracy'].append(acc)
            all_metrics['precision'].append(prec)
            all_metrics['recall'].append(rec)
            all_metrics['f1'].append(f1)

            print(f"Fold {fold+1}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

        print("-" * 60)
        print(f"Average: Acc={np.mean(all_metrics['accuracy']):.3f}, "
              f"F1={np.mean(all_metrics['f1']):.3f}")

        return all_metrics

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['SELL (0)', 'BUY (1)', 'NEUTRAL (2)'],
                                    zero_division=0))

        # Trading-specific metrics
        # Focus on BUY (1) and SELL (0) predictions, ignore NEUTRAL
        trade_mask = (y_pred != 2)
        if trade_mask.sum() > 0:
            trade_pred = y_pred[trade_mask]
            trade_true = y_test[trade_mask]
            trade_correct = (trade_pred == trade_true).sum()
            trade_acc = trade_correct / len(trade_pred)
            print(f"\nTrading Accuracy (excluding NEUTRAL): {trade_acc:.4f}")
            print(f"Trade signals generated: {len(trade_pred)}")
            metrics['trading_accuracy'] = trade_acc
            metrics['num_signals'] = len(trade_pred)

        return metrics

    def _print_feature_importance(self, top_n: int = 20):
        """Print top N important features."""
        if not hasattr(self.model, 'feature_importances_'):
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n" + "=" * 60)
        print(f"TOP {top_n} FEATURE IMPORTANCE")
        print("=" * 60)

        for i in range(min(top_n, len(self.feature_names))):
            idx = indices[i]
            print(f"{i+1:2}. {self.feature_names[idx]:<30} {importances[idx]:.4f}")

    def save_model(self, name: str = None):
        """Save trained model and scaler."""
        if self.model is None:
            print("No model to save!")
            return

        name = name or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_path = os.path.join(self.model_dir, f"{name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"{name}_scaler.joblib")
        meta_path = os.path.join(self.model_dir, f"{name}_meta.joblib")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'feature_names': self.feature_names,
            'all_feature_names': self.all_feature_names,
            'feature_selector': self.feature_selector,
            'metrics': self.training_metrics
        }, meta_path)

        print(f"\n✓ Model saved to: {model_path}")

    def load_model(self, name: str):
        """Load a saved model."""
        model_path = os.path.join(self.model_dir, f"{name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"{name}_scaler.joblib")
        meta_path = os.path.join(self.model_dir, f"{name}_meta.joblib")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        meta = joblib.load(meta_path)
        self.feature_names = meta['feature_names']
        self.all_feature_names = meta.get('all_feature_names', [])
        self.feature_selector = meta.get('feature_selector', None)
        self.training_metrics = meta['metrics']

        print(f"✓ Model loaded: {name}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training accuracy: {self.training_metrics.get('accuracy', 0):.4f}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Note: Features should already be the 50 selected features (from self.feature_names)

        Returns:
            Tuple of (predictions, probabilities)
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
