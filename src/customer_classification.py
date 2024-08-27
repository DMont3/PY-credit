import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Imprimir informações sobre valores nulos
    print("\nValores nulos por coluna:")
    print(X.isnull().sum())

    # Remover a coluna alvo (Cluster) das features, se existir
    if 'Cluster' in X.columns:
        y = X['Cluster']
        X = X.drop('Cluster', axis=1)
    else:
        y = None
        print("Aviso: Coluna 'Cluster' não encontrada. Não é possível realizar a classificação.")

    # Usar SimpleImputer para lidar com valores NaN
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_imputed)

    return X_escalado, y, X.columns.tolist()


def treinar_classificador(X, y):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_treino, y_treino)

    y_pred = clf.predict(X_teste)

    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, y_pred))

    # Matriz de Confusão
    cm = confusion_matrix(y_teste, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    salvar_figura('matriz_confusao.png')

    # Salvar o modelo treinado
    joblib.dump(clf, 'customer_classifier.joblib')

    return clf


def main():
    # Carregar e preprocessar os dados
    df = carregar_dados('data/processed/segmented_data.csv')
    X_escalado, y, features = preprocessar_dados(df)

    if y is not None:
        # Treinar o modelo de classificação
        classificador = treinar_classificador(X_escalado, y)

        print("\nFeatures usadas no modelo:", features)
        print("\nClassificação de novos clientes concluída.")
        print("Modelo de classificação salvo como 'customer_classifier.joblib'")
    else:
        print("Não foi possível treinar o modelo devido à falta da variável alvo (Cluster).")


if __name__ == "__main__":
    main()