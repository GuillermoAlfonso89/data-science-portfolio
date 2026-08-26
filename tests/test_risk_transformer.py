import numpy as np

from risk_transformer import RiskLoadingTransformer


def test_fit_calcula_percentil_90():
    sev = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    transformer = RiskLoadingTransformer()
    transformer.fit(sev)
    assert transformer.umbral_ == np.percentile(sev, 90)

def test_transform_aplica_recargo_correctamente():
    sev = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    primas = [50, 95]  # una prima claramente bajo el umbral, otra claramente sobre
    
    transformer = RiskLoadingTransformer(recargo_pct=0.20)
    transformer.fit(sev)
    resultado = transformer.transform(primas)
    
    assert resultado[0] == primas[0]
    assert resultado[1] == 95 * 1.20