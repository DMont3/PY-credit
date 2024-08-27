import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from utils import salvar_figura

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

    # Usar SimpleImputer para lidar com valores NaN
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Normalizar as features
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_imputed)

    return X_escalado, colunas_numericas.tolist()

def encontrar_numero_otimo_clusters(X, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Para n_clusters = {n_clusters}, o score de silhouette é : {silhouette_avg}")

    # Plotar os scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bx-')
    plt.xlabel('Número de clusters')
    plt.ylabel('Score de Silhouette')
    plt.title('Análise de Silhouette para o k ideal')
    salvar_figura('analise_silhouette.png')

    return silhouette_scores.index(max(silhouette_scores)) + 2

def segmentar_clientes(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels

def analisar_segmentos(df, cluster_labels, features):
    df['Cluster'] = cluster_labels
    analise_segmentos = df.groupby('Cluster')[features].mean()
    print("\nAnálise dos Segmentos:")
    print(analise_segmentos)

    # Visualizar segmentos (usando as duas primeiras features)
    if len(features) >= 2:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('Segmentos de Clientes')
        plt.colorbar(scatter)
        salvar_figura('segmentos_clientes.png')
    else:
        print("Não há features suficientes para criar o gráfico de dispersão.")

def main():
    # Carregar e preprocessar os dados
    df = carregar_dados('data/raw/sample_data.csv')
    X_escalado, features = preprocessar_dados(df)

    print("Colunas usadas para segmentação:", features)

    # Encontrar o número ótimo de clusters
    n_clusters = encontrar_numero_otimo_clusters(X_escalado)
    print(f"\nNúmero ótimo de clusters: {n_clusters}")

    # Segmentar clientes
    cluster_labels = segmentar_clientes(X_escalado, n_clusters)

    # Analisar os segmentos
    analisar_segmentos(df, cluster_labels, features)

    # Adicionar os labels de cluster ao dataframe original
    df['Cluster'] = cluster_labels

    # Construir o caminho do diretório 'data/processed'
    diretorio_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(diretorio_projeto, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Salvando arquivo em: {processed_dir}")

    # Salvar os dados segmentados
    df.to_csv(os.path.join(processed_dir, 'segmented_data.csv'), index=False)
    print("\nSegmentação de clientes concluída. Dados segmentados salvos.")

if __name__ == "__main__":
    main()