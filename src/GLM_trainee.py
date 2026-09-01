import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from simulate_data import generar_dataset


def preparar_X(data, columnas_referencia=None):
    X = pd.get_dummies(data[['edad', 'tipo_vehiculo', 'antiguedad_licencia']],
                        columns=['tipo_vehiculo'], drop_first=True)
    X = X.astype(float)
    X = sm.add_constant(X)
    return X


def main():
    df = generar_dataset()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = preparar_X(train_df)
    y_train = train_df['severidad']
    X_test = preparar_X(test_df)
    y_test = test_df['severidad']
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    modelo = sm.GLM(y_train, X_train, family=sm.families.Gamma(link=sm.families.links.Log()))
    resultado = modelo.fit()

    print(resultado.summary())

    predicciones_test = resultado.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones_test))

    print(f"\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | Promedio real: {y_test.mean():.2f}")


if __name__ == "__main__":
    main()