# ML Trading Module
"""
Machine learning based trading strategies.
"""

# Use lazy imports to avoid circular dependencies
from .data_collector import DataCollector
from .feature_engineer import FeatureEngineer
from .trainer import MLTrainer

# MLStrategy imports BaseStrategy which may have dependencies
# Import it lazily when needed
def get_ml_strategy():
    from .ml_strategy import MLStrategy
    return MLStrategy

__all__ = [
    'DataCollector',
    'FeatureEngineer',
    'MLTrainer',
    'get_ml_strategy'
]
