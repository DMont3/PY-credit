import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from utils import salvar_figura
import joblib


def carregar_dados(caminho_relativo):
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    diretorio_projeto = os.path.dirname(diretorio_atual)
    caminho_completo = os.path.join(diretorio_projeto, caminho_relativo)
    return pd.read_csv(caminho_completo)


def preprocessar_dados(df):
    # Selecionar apenas colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    X = df[colunas_numericas]

    # Remover a coluna alvo (valorAprovado) das features, se existir
    if 'valorAprovado' in X.columns:
        y = X['valorAprovado']
        X = X.drop('valorAprovado', axis=1)

        # Imputar valores NaN em y com a mediana, ou remover registros com NaN
        y = y.fillna(y.median())
    else:
        y = None
        print("Aviso: Coluna 'valorAprovado' não encontrada. Usando todas as colunas numéricas como features.")

    # Usar SimpleImputer para lidar com valores NaN em X
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_imputed)

    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_poly)

    return X_escalado, y, X.columns.tolist()

def recomendar_limite_credito(modelo, dados_cliente):
    valor_previsto = modelo.predict(dados_cliente)
    limite_maximo = min(valor_previsto[0], 50000)  # Exemplo de limite máximo de $50,000
    return limite_maximo


def treinar_modelo(X, y):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    gbr = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_treino, y_treino)

    print("Melhores parâmetros:", grid_search.best_params_)
    print("Melhor score:", -grid_search.best_score_)

    melhor_modelo = grid_search.best_estimator_

    # Salvar o modelo treinado
    joblib.dump(melhor_modelo, 'credit_recommendation_model.joblib')

    return melhor_modelo


def main():
    df = carregar_dados('data/processed/segmented_data.csv')
    X_escalado, y, features = preprocessar_dados(df)

    if y is not None:
        if y.isna().sum() > 0:
            print("Há valores NaN na variável alvo após o preprocessamento. Revise os dados.")
            return

        modelo_gbr = treinar_modelo(X_escalado, y)

        print("\nFeatures usadas no modelo:", features)

        # Exemplo de novo cliente (ajuste conforme necessário)
        novo_cliente = np.random.rand(1, X_escalado.shape[1])  # Cria um array aleatório com o mesmo número de features

        limite_recomendado = recomendar_limite_credito(modelo_gbr, novo_cliente)
        print(f"Limite de crédito recomendado: {limite_recomendado}")

        print("Modelo de recomendação de crédito salvo como 'credit_recommendation_model.joblib'")
    else:
        print("Não foi possível treinar o modelo devido à falta da variável alvo (valorAprovado).")


if __name__ == "__main__":
    main()