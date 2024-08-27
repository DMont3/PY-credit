import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import salvar_figura


def carregar_dados(caminho_relativo):
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    diretorio_projeto = os.path.dirname(diretorio_atual)
    caminho_completo = os.path.join(diretorio_projeto, caminho_relativo)
    return pd.read_csv(caminho_completo)


def explorar_dados(df):
    print("Primeiras linhas do dataset:")
    print(df.head())

    print("\nInformações do dataset:")
    print(df.info())

    print("\nEstatísticas descritivas:")
    print(df.describe())

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nTipos de dados:")
    print(df.dtypes)

    identificar_colunas_nao_numericas(df)


def identificar_colunas_nao_numericas(df):
    colunas_nao_numericas = df.select_dtypes(exclude=[np.number]).columns
    if len(colunas_nao_numericas) > 0:
        print("\nColunas não numéricas identificadas:")
        for col in colunas_nao_numericas:
            print(f"- {col}: {df[col].dtype}")
        print("Estas colunas serão excluídas da matriz de correlação.")


def plotar_matriz_correlacao(df):
    # Selecionar apenas colunas numéricas
    df_numerico = df.select_dtypes(include=[np.number])

    if df_numerico.empty:
        print("Não há colunas numéricas para calcular a matriz de correlação.")
        return

    matriz_correlacao = df_numerico.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de Correlação (Apenas Colunas Numéricas)')
    salvar_figura('matriz_correlacao.png')


def plotar_distribuicoes_numericas(df):
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = (len(colunas_numericas) + n_cols - 1) // n_cols

    fig, eixos = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    eixos = eixos.flatten()

    for i, col in enumerate(colunas_numericas):
        sns.histplot(df[col], ax=eixos[i], kde=True)
        eixos[i].set_title(f'Distribuição de {col}')
        eixos[i].set_xlabel('')

    # Remover subplots vazios
    for j in range(i + 1, len(eixos)):
        fig.delaxes(eixos[j])

    plt.tight_layout()
    salvar_figura('distribuicoes_numericas.png')


def plotar_distribuicoes_categoricas(df):
    colunas_categoricas = df.select_dtypes(include=['object']).columns

    for col in colunas_categoricas:
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Contagem de {col}')
        plt.xlabel('')
        plt.ylabel('Contagem')
        plt.xticks(rotation=45)
        salvar_figura(f'{col}_distribuicao.png')


def tratar_valores_faltantes(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)


def remover_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]


def main():
    df = carregar_dados('data/raw/sample_data.csv')
    explorar_dados(df)
    plotar_matriz_correlacao(df)
    plotar_distribuicoes_numericas(df)
    plotar_distribuicoes_categoricas(df)
    tratar_valores_faltantes(df)

    # Remover outliers (se necessário)
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        df = remover_outliers(df, col)

    print("Exploração de dados e pré-processamento concluídos.")


if __name__ == "__main__":
    main()