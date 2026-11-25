"""Compatibility import; canonical implementation is lithiumscope.model_2.diagnostics.spectral_overfit."""
from importlib import import_module
import sys

if __name__ == "__main__":
    import_module("lithiumscope.model_2.diagnostics.spectral_overfit").main()
else:
    sys.modules[__name__] = import_module("lithiumscope.model_2.diagnostics.spectral_overfit")
