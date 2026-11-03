"""Models whose predict() method returns dollars and an occupancy rate.

Training uses log1p(price) because listing prices are skewed. Callers, including
the API, should not have to remember to invert that transform.
"""
import numpy as np


class DollarRegressor:
    """Wrap a fitted regressor so predict() is in the original units."""

    def __init__(self, model, kind):
        self.model = model
        self.kind = kind

    def predict(self, features):
        raw = np.asarray(self.model.predict(features), dtype=float)
        if self.kind == 'price':
            return np.clip(np.expm1(raw), 15, 5000)
        return np.clip(raw, 0, 1)

    def __getattr__(self, name):
        # Unpickling asks for __setstate__ before attributes exist. Looking up
        # self.model through __getattr__ would recurse forever.
        try:
            model = object.__getattribute__(self, 'model')
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(model, name)
