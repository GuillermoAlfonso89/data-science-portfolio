import numpy as np
import pandas as pd


def generar_dataset(n=2000, seed=42):
    """
    Genera un dataset simulado de seguros: severidad de siniestros
    en función de edad, tipo de vehículo y antigüedad de licencia.
    """
    np.random.seed(seed)

    edad = np.random.randint(18, 75, size=n)
    tipo_vehiculo = np.random.choice(['sedan', 'SUV', 'pickup'], size=n, p=[0.5, 0.3, 0.2])
    antiguedad_licencia = np.random.randint(0, 40, size=n)

    base = 1000
    efecto_edad = -5 * edad
    efecto_vehiculo = np.where(tipo_vehiculo == 'SUV', 300,
                        np.where(tipo_vehiculo == 'pickup', 150, 0))
    ruido = np.random.gamma(shape=2, scale=100, size=n)

    severidad = np.clip(base + efecto_edad + efecto_vehiculo + ruido, 100, None)

    return pd.DataFrame({
        'edad': edad,
        'tipo_vehiculo': tipo_vehiculo,
        'antiguedad_licencia': antiguedad_licencia,
        'severidad': severidad
    })