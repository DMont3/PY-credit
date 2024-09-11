# visualizacao_resultados.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    return pd.read_csv(caminho_arquivo)

def configurar_estilo_plot():
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

def salvar_figura(nome_arquivo: str, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_distribuicao_status(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    if 'status_previsto' in df.columns:
        sns.countplot(x='status_previsto', data=df, palette='viridis')
        plt.title('Distribuição dos Status Previstos', fontsize=18)
    else:
        sns.countplot(x='status', data=df, palette='viridis')
        plt.title('Distribuição dos Status', fontsize=18)
    plt.xlabel('Status', fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    salvar_figura('distribuicao_status.png')

def plot_valor_aprovado(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    if 'valor_aprovado_previsto' in df.columns:
        plt.scatter(df['valorAprovado'], df['valor_aprovado_previsto'], alpha=0.5, color='navy')
        plt.title('Valor Aprovado Real vs Previsto', fontsize=18)
        plt.xlabel('Valor Aprovado Real', fontsize=14)
        plt.ylabel('Valor Aprovado Previsto', fontsize=14)
        plt.plot([df['valorAprovado'].min(), df['valorAprovado'].max()],
                 [df['valorAprovado'].min(), df['valorAprovado'].max()],
                 'r--', lw=2)
    else:
        sns.histplot(df['valorAprovado'], bins=50, kde=True, color='navy')
        plt.title('Distribuição do Valor Aprovado', fontsize=18)
        plt.xlabel('Valor Aprovado', fontsize=14)
        plt.ylabel('Frequência', fontsize=14)
    salvar_figura('valor_aprovado.png')

def plot_correlacao_features(df: pd.DataFrame):
    features_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[features_numericas].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', square=True)
    plt.title('Correlação entre Features Numéricas', fontsize=18)
    salvar_figura('correlacao_features.png')

def plot_distribuicao_grupos(df: pd.DataFrame):
    if 'client_group' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='client_group', data=df, palette='Set3')
        plt.title('Distribuição dos Grupos de Clientes', fontsize=18)
        plt.xlabel('Grupo', fontsize=14)
        plt.ylabel('Contagem', fontsize=14)
        salvar_figura('distribuicao_grupos.png')

def plot_valor_aprovado_por_grupo(df: pd.DataFrame):
    if 'client_group' in df.columns and 'valorAprovado' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='client_group', y='valorAprovado', data=df, palette='Set3')
        plt.title('Valor Aprovado por Grupo de Cliente', fontsize=18)
        plt.xlabel('Grupo', fontsize=14)
        plt.ylabel('Valor Aprovado', fontsize=14)
        salvar_figura('valor_aprovado_por_grupo.png')

def main():
    configurar_estilo_plot()
    caminho_arquivo = input("Insira o caminho para o arquivo de dados: ")
    df = carregar_dados(caminho_arquivo)

    plot_distribuicao_status(df)
    plot_valor_aprovado(df)
    plot_correlacao_features(df)
    plot_distribuicao_grupos(df)
    plot_valor_aprovado_por_grupo(df)

    logging.info("Visualizações geradas e salvas.")

if __name__ == "__main__":
    main()