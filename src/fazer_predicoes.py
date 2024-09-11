import pandas as pd
import numpy as np
import joblib
import logging
from typing import Tuple
import os
from preprocessamento import carregar_dados, converter_colunas_data, tratar_valores_ausentes, criar_features_derivadas, definir_colunas_por_tipo, criar_preprocessador


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_model_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def carregar_modelos():
    try:
        classificador = joblib.load(get_model_path('classificador_modelo.joblib'))
        regressor = joblib.load(get_model_path('regressor_modelo.joblib'))
        preprocessor_data = joblib.load(get_model_path('preprocessador_data.joblib'))
        preprocessador = criar_preprocessador(preprocessor_data['colunas_numericas'], preprocessor_data['colunas_categoricas'])
        logging.info("Modelos e dados do preprocessador carregados com sucesso.")
        return classificador, regressor, preprocessador
    except FileNotFoundError as e:
        logging.error(f"Arquivo não encontrado: {e.filename}")
        logging.error("Certifique-se de que os arquivos de modelo estão na mesma pasta que este script.")
        raise


def preprocessar_novos_dados(df, preprocessador):
    df = converter_colunas_data(df)
    df = tratar_valores_ausentes(df)
    df = criar_features_derivadas(df)

    colunas_numericas, colunas_categoricas, colunas_para_remover = definir_colunas_por_tipo(df)
    df_preprocessado = df.drop(columns=colunas_para_remover)

    X = preprocessador.fit_transform(df_preprocessado)
    return X


def fazer_predicoes(X, classificador, regressor, credit_recommender):
    status_predicao = classificador.predict(X)
    valor_aprovado_predicao = regressor.predict(X)
    recomendacoes = [credit_recommender(row.reshape(1, -1)) for row in X]
    return status_predicao, valor_aprovado_predicao, recomendacoes

def salvar_predicoes(df_original, status_predicao, valor_aprovado_predicao, recomendacoes):
    df_resultado = df_original.copy()
    df_resultado['status_previsto'] = status_predicao
    df_resultado['valor_aprovado_previsto'] = valor_aprovado_predicao
    df_resultado['recomendacao_credito'] = [rec[2] for rec in recomendacoes]

    nome_arquivo = 'resultados_predicoes.csv'
    df_resultado.to_csv(nome_arquivo, index=False)
    logging.info(f"Predições salvas em {nome_arquivo}")

def main():
    # Carregar modelos
    try:
        classificador = joblib.load(get_model_path('classificador_modelo.joblib'))
        regressor = joblib.load(get_model_path('regressor_modelo.joblib'))
        credit_recommender = joblib.load(get_model_path('credit_recommender.joblib'))
        preprocessor_data = joblib.load(get_model_path('preprocessador_data.joblib'))
        preprocessador = criar_preprocessador(preprocessor_data['colunas_numericas'], preprocessor_data['colunas_categoricas'])
        logging.info("Modelos, dados do preprocessador e recomendador de crédito carregados com sucesso.")
    except FileNotFoundError as e:
        logging.error(f"Arquivo não encontrado: {e.filename}")
        logging.error("Certifique-se de que os arquivos de modelo estão na mesma pasta que este script.")
        return

    # Carregar novos dados
    caminho_arquivo = input("Por favor, insira o caminho completo para o arquivo CSV com novos dados: ")
    df_novos = carregar_dados(caminho_arquivo)

    # Preprocessar novos dados
    X_novos = preprocessar_novos_dados(df_novos, preprocessador)

    # Fazer predições
    status_predicao, valor_aprovado_predicao, recomendacoes = fazer_predicoes(X_novos, classificador, regressor, credit_recommender)

    # Salvar resultados
    salvar_predicoes(df_novos, status_predicao, valor_aprovado_predicao, recomendacoes)

    # Exibir algumas estatísticas das predições
    logging.info("\nEstatísticas das predições:")
    logging.info(f"Distribuição de status previstos:\n{pd.Series(status_predicao).value_counts(normalize=True)}")
    logging.info(f"\nEstatísticas do valor aprovado previsto:")
    logging.info(f"Média: {np.mean(valor_aprovado_predicao):.2f}")
    logging.info(f"Mediana: {np.median(valor_aprovado_predicao):.2f}")
    logging.info(f"Mínimo: {np.min(valor_aprovado_predicao):.2f}")
    logging.info(f"Máximo: {np.max(valor_aprovado_predicao):.2f}")
    logging.info("\nExemplos de recomendações de crédito:")
    for i in range(min(5, len(recomendacoes))):
        logging.info(f"Cliente {i+1}: {recomendacoes[i][2]}")

if __name__ == "__main__":
    main()