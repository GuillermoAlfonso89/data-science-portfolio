import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RiskLoadingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, recargo_pct=0.15):
        self.recargo_pct = recargo_pct
        self.umbral_ = None

    def fit(self, severidades, y=None):
        self.umbral_ = np.percentile(severidades, 90)
        return self

    def transform(self, primas):
        primas = np.asarray(primas)
        return np.where(primas > self.umbral_, primas * (1 + self.recargo_pct), primas)