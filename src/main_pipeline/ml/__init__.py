"""Optional sklearn filter on detector proposals. Not a learned detector."""

from src.main_pipeline.ml.features import FEATURE_COLUMNS
from src.main_pipeline.ml.infer import apply_ml_filter
from src.main_pipeline.ml.patch_features import PATCH_FEATURE_NAMES
from src.main_pipeline.ml.patches import PATCH_SIZE

__all__ = ["FEATURE_COLUMNS", "PATCH_FEATURE_NAMES", "PATCH_SIZE", "apply_ml_filter"]
