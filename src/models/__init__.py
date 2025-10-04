"""Models package for skin lesion classification."""

from .cnn_models import SkinLesionCNN, create_ensemble_model

__all__ = ['SkinLesionCNN', 'create_ensemble_model']

