import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carregar_dados(caminho_arquivo):
    try:
        df = pd.read_csv(caminho_arquivo, parse_dates=['primeiraCompra', 'dataAprovadoEmComite', 'dataAprovadoNivelAnalista'])
        logging.info(f"Dados carregados com sucesso de {caminho_arquivo}")
        return df
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {caminho_arquivo}")
        raise


def salvar_figura(nome_arquivo, diretorio='figuras'):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    plt.savefig(caminho_completo)
    plt.close()
    logging.info(f"Figura salva: {caminho_completo}")


def analisar_estrutura_dados(df):
    logging.info("Estrutura dos dados:")
    logging.info(f"Número de linhas: {df.shape[0]}")
    logging.info(f"Número de colunas: {df.shape[1]}")
    logging.info("\nTipos de dados:")
    logging.info(df.dtypes)


def analisar_valores_ausentes(df):
    valores_ausentes = df.isnull().sum()
    percentual_ausente = 100 * df.isnull().sum() / len(df)
    tabela_ausentes = pd.concat([valores_ausentes, percentual_ausente], axis=1, keys=['Total', 'Percentual'])
    logging.info("\nValores ausentes:")
    logging.info(tabela_ausentes[tabela_ausentes['Total'] > 0].sort_values('Total', ascending=False))

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Mapa de Calor de Valores Ausentes')
    salvar_figura('valores_ausentes_heatmap.png')


def analisar_distribuicoes(df):
    colunas_numericas = df.select_dtypes(include=[np.number]).columns

    for coluna in colunas_numericas:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[coluna], kde=True)
        plt.title(f'Distribuição de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
        salvar_figura(f'distribuicao_{coluna}.png')


def analisar_correlacoes(df):
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    correlacoes = df[colunas_numericas].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlacoes, annot=False, cmap='coolwarm')
    plt.title('Mapa de Calor das Correlações')
    salvar_figura('correlacoes_heatmap.png')


def analisar_status_aprovacao(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='status', data=df)
    plt.title('Distribuição do Status de Aprovação')
    plt.xlabel('Status')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    salvar_figura('distribuicao_status.png')

    aprovados = df[df['status'].str.contains('Aprovado', na=False)]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='valorSolicitado', y='valorAprovado', data=aprovados)
    plt.title('Valor Solicitado vs Valor Aprovado')
    plt.xlabel('Valor Solicitado')
    plt.ylabel('Valor Aprovado')
    salvar_figura('solicitado_vs_aprovado.png')


def analisar_risco(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='definicaoRisco', y='valorAprovado', data=df)
    plt.title('Valor Aprovado por Definição de Risco')
    plt.xlabel('Definição de Risco')
    plt.ylabel('Valor Aprovado')
    plt.xticks(rotation=45)
    salvar_figura('risco_vs_aprovado.png')


def analisar_tempo_operacao(df):
    df['tempo_operacao'] = pd.Timestamp.now().year - df['anoFundacao']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tempo_operacao', y='valorAprovado', data=df)
    plt.title('Tempo de Operação vs Valor Aprovado')
    plt.xlabel('Tempo de Operação (anos)')
    plt.ylabel('Valor Aprovado')
    salvar_figura('tempo_operacao_vs_aprovado.png')


def explorar_dados(caminho_arquivo):
    df = carregar_dados(caminho_arquivo)

    analisar_estrutura_dados(df)
    analisar_valores_ausentes(df)
    analisar_distribuicoes(df)
    analisar_correlacoes(df)
    analisar_status_aprovacao(df)
    analisar_risco(df)
    analisar_tempo_operacao(df)

    return df


if __name__ == "__main__":
    # Solicitar o caminho do arquivo ao usuário
    caminho_arquivo = input("Por favor, insira o caminho completo para o arquivo CSV: ")

    try:
        df = explorar_dados(caminho_arquivo)
        logging.info("Exploração de dados concluída. Verifique as visualizações geradas na pasta 'figuras'.")
    except FileNotFoundError:
        logging.error("O programa será encerrado devido ao erro no carregamento do arquivo.")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {str(e)}")