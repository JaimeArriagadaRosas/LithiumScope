"""Historical import compatibility for the training-session module."""
import sys
from lithiumscope.core.logging import training

sys.modules[__name__] = training
