"""
ML Exit Model Trainer

Trains a model to predict optimal exit timing for open positions.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger('trading_bot')


class ExitModelTrainer:
    """Trains ML model to predict optimal exit timing."""

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []

    def generate_exit_labels(
        self,
        df: pd.DataFrame,
        entry_signals: np.ndarray,
        entry_directions: np.ndarray,
        max_hold_bars: int = 48,
        profit_target_pips: float = 30,
        stop_loss_pips: float = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate exit labels based on optimal exit points.

        For each entry signal, determines the optimal exit point within max_hold_bars.
        Label = 1 if current bar is within 2 bars of optimal exit.

        Returns:
            exit_labels: Array of 0 (HOLD) or 1 (EXIT_NOW) for each bar
            position_features: Additional features about current position
        """
        n = len(df)
        exit_labels = np.zeros(n)

        # Position tracking features
        bars_in_position = np.zeros(n)
        unrealized_pips = np.zeros(n)
        max_profit_seen = np.zeros(n)
        max_drawdown_seen = np.zeros(n)

        close_prices = df['close'].values

        for i in range(n - max_hold_bars):
            if not entry_signals[i]:
                continue

            direction = entry_directions[i]  # 1 = BUY, 0 = SELL
            entry_price = close_prices[i]

            # Calculate profit at each bar after entry
            profits = []
            for j in range(1, max_hold_bars + 1):
                if i + j >= n:
                    break

                if direction == 1:  # BUY
                    pips = (close_prices[i + j] - entry_price) * 10000
                else:  # SELL
                    pips = (entry_price - close_prices[i + j]) * 10000

                profits.append((j, pips))

            if not profits:
                continue

            # Find optimal exit (maximum profit point)
            optimal_bar, max_profit = max(profits, key=lambda x: x[1])

            # Mark bars near optimal exit as EXIT_NOW
            optimal_idx = i + optimal_bar
            for offset in range(-2, 3):  # 2 bars before/after optimal
                target_idx = optimal_idx + offset
                if 0 <= target_idx < n:
                    exit_labels[target_idx] = 1

            # Also mark if hit profit target or stop loss
            for j, pips in profits:
                target_idx = i + j
                if target_idx < n:
                    if pips >= profit_target_pips or pips <= -stop_loss_pips:
                        exit_labels[target_idx] = 1
                        break

            # Fill position features for bars after entry
            running_max = 0
            running_min = 0
            for j in range(1, min(max_hold_bars + 1, n - i)):
                target_idx = i + j

                if direction == 1:
                    pips = (close_prices[target_idx] - entry_price) * 10000
                else:
                    pips = (entry_price - close_prices[target_idx]) * 10000

                running_max = max(running_max, pips)
                running_min = min(running_min, pips)

                bars_in_position[target_idx] = j
                unrealized_pips[target_idx] = pips
                max_profit_seen[target_idx] = running_max
                max_drawdown_seen[target_idx] = running_min

        position_features = np.column_stack([
            bars_in_position,
            unrealized_pips,
            max_profit_seen,
            max_drawdown_seen,
            unrealized_pips - max_profit_seen,  # Pullback from peak
            bars_in_position / max_hold_bars,   # Normalized time
        ])

        return exit_labels, position_features

    def create_exit_features(
        self,
        market_features: pd.DataFrame,
        position_features: np.ndarray
    ) -> np.ndarray:
        """Combine market features with position features."""
        market_array = market_features.values
        combined = np.hstack([market_array, position_features])
        return combined

    def train_exit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        top_features: int = 40
    ) -> Dict[str, Any]:
        """Train ensemble exit model."""

        # Filter to only bars with position (non-zero position features)
        has_position = X[:, -1] > 0  # bars_in_position > 0
        X_pos = X[has_position]
        y_pos = y[has_position]

        print(f"Training samples with positions: {len(X_pos)}")
        print(f"Exit signals: {y_pos.sum():.0f} ({y_pos.mean()*100:.1f}%)")

        # Handle NaN
        X_pos = np.nan_to_num(X_pos, nan=0.0, posinf=0.0, neginf=0.0)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pos, y_pos, test_size=0.2, random_state=42, stratify=y_pos
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(top_features, X_train.shape[1]))
        X_train = self.feature_selector.fit_transform(X_train, y_train)
        X_test = self.feature_selector.transform(X_test)

        selected_mask = self.feature_selector.get_support()
        self.feature_names = [f for f, m in zip(feature_names, selected_mask) if m]

        print(f"Selected {len(self.feature_names)} features")

        # Train models
        print("\n[1/2] Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=1  # Set to 1 to avoid overhead in backtest loops
        )
        rf.fit(X_train, y_train)
        rf_acc = rf.score(X_test, y_test)
        print(f"  Accuracy: {rf_acc:.4f}")

        print("[2/2] Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_train, y_train)
        gb_acc = gb.score(X_test, y_test)
        print(f"  Accuracy: {gb_acc:.4f}")

        self.models = {
            'random_forest': rf,
            'gradient_boosting': gb
        }

        # Ensemble predictions
        rf_proba = rf.predict_proba(X_test)[:, 1]
        gb_proba = gb.predict_proba(X_test)[:, 1]
        avg_proba = (rf_proba + gb_proba) / 2

        # High confidence threshold
        high_conf_mask = avg_proba > 0.7
        if high_conf_mask.sum() > 0:
            high_conf_acc = (y_test[high_conf_mask] == 1).mean()
            print(f"\nHigh confidence exits (>70%): {high_conf_mask.sum()} samples, {high_conf_acc:.2%} precision")

        return {
            'rf_accuracy': rf_acc,
            'gb_accuracy': gb_acc,
            'samples': len(X_pos)
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict exit signals.

        Returns:
            predictions: 1 = EXIT_NOW, 0 = HOLD
            probabilities: Confidence of exit
        """
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        X = self.feature_selector.transform(X)

        rf_proba = self.models['random_forest'].predict_proba(X)[:, 1]
        gb_proba = self.models['gradient_boosting'].predict_proba(X)[:, 1]

        avg_proba = (rf_proba + gb_proba) / 2
        predictions = (avg_proba > 0.6).astype(int)

        return predictions, avg_proba

    def save_model(self, name: str) -> None:
        """Save exit model."""
        path = self.model_dir / f'exit_model_{name}.joblib'
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names
        }, path)
        print(f"✓ Exit model saved: {path}")

    def load_model(self, name: str) -> None:
        """Load exit model."""
        path = self.model_dir / f'exit_model_{name}.joblib'
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_selector = data['feature_selector']
        self.feature_names = data['feature_names']

        # FORCE n_jobs=1 to avoid library conflicts in backtest loops
        for name, model in self.models.items():
            if hasattr(model, 'n_jobs'):
                model.n_jobs = 1

        print(f"✓ Exit model loaded: {name}")
