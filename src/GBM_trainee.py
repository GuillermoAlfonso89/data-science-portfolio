import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from simulate_data import generar_dataset


def main():
    df = generar_dataset()

    X = pd.get_dummies(df[['edad', 'tipo_vehiculo', 'antiguedad_licencia']],
                        columns=['tipo_vehiculo'], drop_first=True).astype(float)
    y = df['severidad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    modelo.fit(X_train, y_train)

    rmse_train = np.sqrt(mean_squared_error(y_train, modelo.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, modelo.predict(X_test)))

    print(f"RMSE train: {rmse_train:.2f}")
    print(f"RMSE test:  {rmse_test:.2f}")
    print(f"Brecha: {rmse_test - rmse_train:.2f}")

    importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
    print("\nFeature importances:")
    print(importancias.sort_values(ascending=False))


if __name__ == "__main__":
    main()