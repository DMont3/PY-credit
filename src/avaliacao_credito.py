# Importar bibliotecas necessárias
import os  # Para manipulação de caminhos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime

# Bibliotecas de Machine Learning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Suprimir avisos
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
sns.set(style='whitegrid')

# Determinar o diretório do script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir caminhos para data e figures fora da pasta src
data_dir = os.path.join(script_dir, '..', 'data')
figures_dir = os.path.join(script_dir, '..', 'figures')

# Criar as pastas 'data' e 'figures' se não existirem
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Caminho completo para o arquivo CSV de entrada
input_csv_path = os.path.join(data_dir, 'sample_data.csv')

# Carregar o dataset
df = pd.read_csv(input_csv_path, encoding='utf-8')  # Caminho ajustado para a pasta 'data'

# Exibir as primeiras linhas
print("Primeiras linhas do dataset:")
print(df.head())

# Exibir informações básicas
print("\nInformações do dataset:")
print(df.info())

# Copiar o DataFrame para evitar modificar os dados originais
dados = df.copy()

# Remover colunas irrelevantes
colunas_remover = [
    'numero_solicitacao', 'razaoSocial', 'nomeFantasia', 'cnpjSemTraco',
    'dataAprovadoEmComite', 'dashboardCorrelacao', 'diferencaPercentualRisco'
]
dados = dados.drop(columns=colunas_remover, errors='ignore')

# Tratar valores nulos na coluna 'valorAprovado'
dados = dados.dropna(subset=['valorAprovado'])  # Remover linhas onde 'valorAprovado' está ausente

# Converter colunas de data para datetime
colunas_data = ['primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite', 'periodoBalanco']
for col in colunas_data:
    if col in dados.columns:
        dados[col] = pd.to_datetime(dados[col], errors='coerce')

# Engenharia de atributos: criar novas features
ano_atual = datetime.today().year
dados['idade_empresa'] = ano_atual - dados['anoFundacao']
dados['dias_desde_primeira_compra'] = (datetime.today() - dados['primeiraCompra']).dt.days

# Tratar variáveis categóricas usando um LabelEncoder por coluna
colunas_categoricas = ['status', 'definicaoRisco', 'intervaloFundacao']
label_encoders = {}
for col in colunas_categoricas:
    if col in dados.columns:
        le = LabelEncoder()
        dados[col] = le.fit_transform(dados[col].astype(str))
        label_encoders[col] = le  # Salvar o LabelEncoder para uso futuro

# Converter colunas booleanas que estão como strings
colunas_booleanas = ['restricoes', 'empresa_MeEppMei']
for col in colunas_booleanas:
    if col in dados.columns:
        dados[col] = dados[col].astype(str).map({'False': 0, 'True': 1, 'nan': 0})

# Remover colunas que não são mais necessárias
colunas_remover = [
    'primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite',
    'periodoBalanco', 'anoFundacao'
]
dados = dados.drop(columns=colunas_remover, errors='ignore')

# Tratar valores nulos restantes usando imputação com a mediana
imputer = SimpleImputer(strategy='median')
dados = pd.DataFrame(imputer.fit_transform(dados), columns=dados.columns)

# Verificar tipos de dados antes do escalonamento
print("\nTipos de dados antes do escalonamento:")
print(dados.dtypes)

# Identificar colunas não numéricas restantes
colunas_nao_numericas = dados.select_dtypes(include=['object']).columns
print("\nColunas não numéricas:")
print(colunas_nao_numericas)

# Remover colunas não numéricas que não foram tratadas
dados = dados.drop(columns=colunas_nao_numericas, errors='ignore')

# Escalonamento de features
scaler = StandardScaler()
features = dados.drop('valorAprovado', axis=1).columns
dados[features] = scaler.fit_transform(dados[features])

# Exibir o DataFrame processado
print("\nDataFrame processado:")
print(dados.head())

# Objetivo 1: Segmentação de Clientes

# Preparar dados para clustering
X_cluster = dados.drop('valorAprovado', axis=1)

# Escolher o número de clusters usando o Método do Cotovelo
inercia = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inercia.append(kmeans.inertia_)

# Plotar a Curva do Cotovelo com nomes em português
plt.figure(figsize=(8, 4))
plt.plot(K, inercia, 'bx-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Determinar k Ótimo')
cotovelo_path = os.path.join(figures_dir, 'metodo_cotovelo.png')
plt.savefig(cotovelo_path)  # Salva na pasta 'figures'
plt.show()

# Escolher k
k_otimo = 5
kmeans = KMeans(n_clusters=k_otimo, random_state=42)
dados['Grupo'] = kmeans.fit_predict(X_cluster)

# Visualizar clusters usando PCA
pca = PCA(n_components=2)
componentes_principais = pca.fit_transform(X_cluster)
pc_df = pd.DataFrame(data=componentes_principais, columns=['PCA 1', 'PCA 2'])  # Renomeado para clareza
pc_df['Grupo'] = dados['Grupo']

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA 1', y='PCA 2', hue='Grupo', data=pc_df, palette='Set1')
plt.title('Segmentação de Clientes (PCA)')
segmentacao_path = os.path.join(figures_dir, 'segmentacao_clientes.png')
plt.savefig(segmentacao_path)  # Salva na pasta 'figures'
plt.show()

# Gráfico de barras mostrando a quantidade de clientes em cada grupo
plt.figure(figsize=(8, 6))
sns.countplot(x='Grupo', data=dados, palette='Set2')
plt.xlabel('Grupo de Clientes')
plt.ylabel('Quantidade de Clientes')
plt.title('Distribuição de Clientes por Grupo')
distribuicao_grupos_path = os.path.join(figures_dir, 'distribuicao_grupos.png')
plt.savefig(distribuicao_grupos_path)  # Salva na pasta 'figures'
plt.show()

# Analisar características dos grupos
resumo_grupos = dados.groupby('Grupo').mean()
print("\nResumo dos Grupos:")
print(resumo_grupos)

# Gerar um arquivo CSV com os grupos
df_grupos = df.copy()
df_grupos['Grupo'] = dados['Grupo']
clientes_segmentados_path = os.path.join(data_dir, 'clientes_segmentados.csv')
df_grupos.to_csv(clientes_segmentados_path, index=False)  # Salva na pasta 'data'

# Objetivo 2: Classificação de Novos Clientes

# Preparar dados para classificação
X = dados.drop(['Grupo', 'valorAprovado'], axis=1)
y = dados['Grupo']

# Dividir o conjunto de dados
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Construir um pipeline de classificação
features_numericas = X.select_dtypes(include=['float64', 'int64']).columns
transformador_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessador = ColumnTransformer(transformers=[
    ('num', transformador_numerico, features_numericas)
])

pipeline_clf = Pipeline(steps=[
    ('preprocessor', preprocessador),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Treinar o pipeline
pipeline_clf.fit(X_treino, y_treino)

# Prever e avaliar
y_pred = pipeline_clf.predict(X_teste)
print("\nRelatório de Classificação:")
print(classification_report(y_teste, y_pred))

# Matriz de Confusão com rótulos em português
matriz_confusao = confusion_matrix(y_teste, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
matriz_confusao_path = os.path.join(figures_dir, 'matriz_confusao.png')
plt.savefig(matriz_confusao_path)  # Salva na pasta 'figures'
plt.show()

# 3. Gráfico de Importância das Features (Substituindo a Correlação)

# Obter o modelo treinado do pipeline_clf
modelo_rf = pipeline_clf.named_steps['classifier']

# Obter as importâncias das features
importancias = modelo_rf.feature_importances_
nomes_features = X.columns

# Criar um DataFrame para as importâncias
df_importancias = pd.DataFrame({
    'Feature': nomes_features,
    'Importancia': importancias
})

# Ordenar o DataFrame por importância
df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)

# Plotar o gráfico de importância das features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importancia', y='Feature', data=df_importancias, palette='viridis')
plt.xlabel('Importância da Feature')
plt.ylabel('Feature')
plt.title('Importância das Features no Modelo de Classificação')
plt.tight_layout()
importancia_features_path = os.path.join(figures_dir, 'importancia_features.png')
plt.savefig(importancia_features_path)  # Salva na pasta 'figures'
plt.show()

# Objetivo 3: Recomendação Automatizada de Crédito

# Classificar as recomendações em categorias
def categorizar_limite_credito(valor):
    if valor <= 10000:
        return 'Baixo'
    elif valor <= 50000:
        return 'Médio'
    elif valor <= 100000:
        return 'Alto'
    else:
        return 'Muito Alto'

# Aplicar a função de categorização aos dados
dados['Categoria_Limite_Credito'] = dados['valorAprovado'].apply(categorizar_limite_credito)

# Verificar se a coluna foi criada
print("Colunas do DataFrame 'dados' após criar 'Categoria_Limite_Credito':")
print(dados.columns)

# Visualização: Distribuição das Categorias de Limite de Crédito com Ordem Específica

plt.figure(figsize=(8, 6))
categorias_ordem = ['Baixo', 'Médio', 'Alto', 'Muito Alto']  # Ordem especificada
sns.countplot(x='Categoria_Limite_Credito', data=dados, palette='Set2', order=categorias_ordem)
plt.xlabel('Categoria do Limite de Crédito')
plt.ylabel('Quantidade de Clientes')
plt.title('Distribuição das Categorias de Limite de Crédito')
distribuicao_categorias_limite_path = os.path.join(figures_dir, 'distribuicao_categorias_limite.png')
plt.savefig(distribuicao_categorias_limite_path)  # Salva na pasta 'figures'
plt.show()

# Fazer uma cópia do DataFrame para evitar modificar 'dados'
dados_reg = dados.copy()

# Preparar dados para regressão
X_reg = dados_reg.drop(['valorAprovado', 'Grupo', 'Categoria_Limite_Credito'], axis=1)
y_reg = dados_reg['valorAprovado']

# Dividir o conjunto de dados
X_treino_reg, X_teste_reg, y_treino_reg, y_teste_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Construir um pipeline de regressão usando RandomForestRegressor
features_numericas_reg = X_reg.select_dtypes(include=['float64', 'int64']).columns
transformador_numerico_reg = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessador_reg = ColumnTransformer(
    transformers=[('num', transformador_numerico_reg, features_numericas_reg)]
)

pipeline_reg = Pipeline(
    steps=[('preprocessor', preprocessador_reg), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))]
)

# Treinar o pipeline
pipeline_reg.fit(X_treino_reg, y_treino_reg)

# Prever e avaliar
y_pred_reg = pipeline_reg.predict(X_teste_reg)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_teste_reg, y_pred_reg)
r2 = r2_score(y_teste_reg, y_pred_reg)
print(f"\nErro Médio Quadrático (MSE): {mse}")
print(f"Coeficiente de Determinação (R²): {r2}")

# Plotar valores reais vs previstos com rótulos em português
plt.figure(figsize=(8, 6))
plt.scatter(y_teste_reg, y_pred_reg, alpha=0.5, color='purple')
plt.plot(
    [y_teste_reg.min(), y_teste_reg.max()],
    [y_teste_reg.min(), y_teste_reg.max()],
    'r--'
)
plt.xlabel('Valor Aprovado Real (R$)')
plt.ylabel('Valor Aprovado Previsto (R$)')
plt.title('Comparação entre Valores Reais e Previstos')
regressao_credito_path = os.path.join(figures_dir, 'regressao_credito.png')
plt.savefig(regressao_credito_path)  # Salva na pasta 'figures'
plt.show()

# Função para classificar novos clientes
def classificar_novo_cliente(novo_cliente):
    # Converter o dicionário para DataFrame
    novo_cliente_df = pd.DataFrame([novo_cliente])

    # Tratar variáveis categóricas usando os LabelEncoders correspondentes
    for col in colunas_categoricas:
        if col in novo_cliente_df.columns:
            le = label_encoders[col]
            novo_cliente_df[col] = le.transform(novo_cliente_df[col].astype(str))

    # Converter colunas booleanas que estão como strings
    for col in colunas_booleanas:
        if col in novo_cliente_df.columns:
            novo_cliente_df[col] = novo_cliente_df[col].astype(str).map({'False': 0, 'True': 1, 'nan': 0})

    # Engenharia de atributos
    novo_cliente_df['idade_empresa'] = ano_atual - novo_cliente_df['anoFundacao']
    novo_cliente_df['dias_desde_primeira_compra'] = (datetime.today() - pd.to_datetime(novo_cliente_df['primeiraCompra'])).dt.days

    # Remover colunas não usadas
    colunas_remover = ['primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite', 'periodoBalanco', 'anoFundacao']
    novo_cliente_df = novo_cliente_df.drop(columns=colunas_remover, errors='ignore')

    # Garantir que as colunas estejam na mesma ordem
    novo_cliente_df = novo_cliente_df[X_treino.columns]

    # Prever grupo usando pipeline_clf
    grupo_label = pipeline_clf.predict(novo_cliente_df)
    return grupo_label[0]

# Função para recomendar limite de crédito
def recomendar_limite_credito(dados_cliente):
    # Converter o dicionário para DataFrame
    cliente_df = pd.DataFrame([dados_cliente])

    # Tratar variáveis categóricas usando os LabelEncoders correspondentes
    for col in colunas_categoricas:
        if col in cliente_df.columns:
            le = label_encoders[col]
            cliente_df[col] = le.transform(cliente_df[col].astype(str))

    # Converter colunas booleanas que estão como strings
    for col in colunas_booleanas:
        if col in cliente_df.columns:
            cliente_df[col] = cliente_df[col].astype(str).map({'False': 0, 'True': 1, 'nan': 0})

    # Engenharia de atributos
    cliente_df['idade_empresa'] = ano_atual - cliente_df['anoFundacao']
    cliente_df['dias_desde_primeira_compra'] = (datetime.today() - pd.to_datetime(cliente_df['primeiraCompra'])).dt.days

    # Remover colunas não usadas
    colunas_remover = ['primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite', 'periodoBalanco', 'anoFundacao']
    cliente_df = cliente_df.drop(columns=colunas_remover, errors='ignore')

    # Garantir que as colunas estejam na mesma ordem
    cliente_df = cliente_df[X_reg.columns]

    # Prever limite de crédito usando pipeline_reg
    limite_previsto = pipeline_reg.predict(cliente_df)[0]

    # Aplicar limites realistas
    limite_previsto = max(0, min(limite_previsto, 150000))  # Limitar entre 0 e 150.000 Reais

    return limite_previsto

# Gerar um arquivo CSV com as recomendações
df_recomendacoes = df.copy()
df_recomendacoes['Grupo'] = dados['Grupo']
df_recomendacoes['Limite_Recomendado'] = dados['valorAprovado']
df_recomendacoes['Categoria_Limite_Credito'] = dados['Categoria_Limite_Credito']
clientes_recomendacoes_path = os.path.join(data_dir, 'clientes_recomendacoes.csv')
df_recomendacoes.to_csv(clientes_recomendacoes_path, index=False)  # Salva na pasta 'data'

# Visualizações Adicionais

# 1. Distribuição dos Limites de Crédito Aprovados com Binning Dinâmico

plt.figure(figsize=(12, 6))

# Definir o limite máximo
limite_max = 150000  # Ajuste conforme a necessidade

# Filtrar os dados para remover valores extremamente altos que podem distorcer o gráfico
dados_filtrados = dados[dados['valorAprovado'] <= limite_max]

# Criar o histograma com binning dinâmico
sns.histplot(dados_filtrados['valorAprovado'], bins='auto', kde=False, color='blue')

# Configurar os rótulos e título em português
plt.xlabel('Valor Aprovado (R$)')
plt.ylabel('Frequência')
plt.title('Distribuição dos Limites de Crédito Aprovados (Valores até R$150.000)')

# Rotacionar os ticks do eixo X em 45 graus para melhor legibilidade
plt.xticks(rotation=45)

# Adicionar grid para melhor visualização
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Salvar o gráfico na pasta 'figures'
distribuicao_limites_path = os.path.join(figures_dir, 'distribuicao_limites.png')
plt.savefig(distribuicao_limites_path)  # Salva na pasta 'figures'
plt.show()

# Exemplo de uso das funções de classificação e recomendação

# Dados de um novo cliente
novo_cliente_dados = {
    'maiorAtraso': 10,
    'margemBrutaAcumulada': 0.5,
    'percentualProtestos': 0.1,
    'prazoMedioRecebimentoVendas': 30,
    'titulosEmAberto': 5000.0,
    'valorSolicitado': 200000.0,
    'status': 'AprovadoAnalista',
    'definicaoRisco': 'De 11 a 30 % - Baixo',
    'percentualRisco': 0.2,
    'ativoCirculante': 150000.0,
    'passivoCirculante': 100000.0,
    'totalAtivo': 200000.0,
    'totalPatrimonioLiquido': 50000.0,
    'endividamento': 50000.0,
    'duplicatasAReceber': 30000.0,
    'estoque': 20000.0,
    'faturamentoBruto': 400000.0,
    'margemBruta': 80000.0,
    'periodoDemonstrativoEmMeses': 12,
    'custos': 320000.0,
    'intervaloFundacao': 'De 6 a 10 anos',
    'capitalSocial': 100000.0,
    'restricoes': 'False',
    'empresa_MeEppMei': 'True',
    'scorePontualidade': 0.8,
    'limiteEmpresaAnaliseCredito': 20000.0,
    'primeiraCompra': '2018-05-10',
    'anoFundacao': 2010
}

# Classificar o novo cliente
grupo_novo_cliente = classificar_novo_cliente(novo_cliente_dados)
print(f"O novo cliente pertence ao Grupo: {grupo_novo_cliente}")

# Recomendar limite de crédito para o novo cliente
limite_novo_cliente = recomendar_limite_credito(novo_cliente_dados)
categoria_limite = categorizar_limite_credito(limite_novo_cliente)
print(f"Limite de Crédito Recomendado: R$ {limite_novo_cliente:,.2f} - Categoria: {categoria_limite}")

# Adição da GUI
def criar_gui():
    # Inicializar a janela principal
    janela = tk.Tk()
    janela.title("Previsão de Limite de Crédito")
    janela.geometry("600x1000")
    janela.resizable(False, False)

    # Título da aplicação
    titulo = ttk.Label(janela, text="Recomendação de Limite de Crédito", font=("Arial", 16))
    titulo.pack(pady=10)

    # Frame para os dados já escritos
    frame_dados = ttk.LabelFrame(janela, text="Dados do Cliente")
    frame_dados.pack(padx=10, pady=10, fill="both", expand=True)

    # Exemplo de dados (pode ser adaptado conforme necessário)
    labels = [
        "Maior Atraso:",
        "Margem Bruta Acumulada:",
        "Percentual Protestos:",
        "Prazo Médio Recebimento Vendas:",
        "Títulos em Aberto:",
        "Valor Solicitado:",
        "Status:",
        "Definição de Risco:",
        "Percentual de Risco:",
        "Ativo Circulante:",
        "Passivo Circulante:",
        "Total Ativo:",
        "Total Patrimônio Líquido:",
        "Endividamento:",
        "Duplicatas a Receber:",
        "Estoque:",
        "Faturamento Bruto:",
        "Margem Bruta:",
        "Período Demonstrativo (meses):",
        "Custos:",
        "Intervalo de Fundação:",
        "Capital Social:",
        "Restrições:",
        "Empresa Me/EPP/MEI:",
        "Score Pontualidade:",
        "Limite Empresa Análise de Crédito:",
        "Primeira Compra:",
        "Ano de Fundação:"
    ]

    valores = [
        novo_cliente_dados['maiorAtraso'],
        novo_cliente_dados['margemBrutaAcumulada'],
        novo_cliente_dados['percentualProtestos'],
        novo_cliente_dados['prazoMedioRecebimentoVendas'],
        novo_cliente_dados['titulosEmAberto'],
        novo_cliente_dados['valorSolicitado'],
        novo_cliente_dados['status'],
        novo_cliente_dados['definicaoRisco'],
        novo_cliente_dados['percentualRisco'],
        novo_cliente_dados['ativoCirculante'],
        novo_cliente_dados['passivoCirculante'],
        novo_cliente_dados['totalAtivo'],
        novo_cliente_dados['totalPatrimonioLiquido'],
        novo_cliente_dados['endividamento'],
        novo_cliente_dados['duplicatasAReceber'],
        novo_cliente_dados['estoque'],
        novo_cliente_dados['faturamentoBruto'],
        novo_cliente_dados['margemBruta'],
        novo_cliente_dados['periodoDemonstrativoEmMeses'],
        novo_cliente_dados['custos'],
        novo_cliente_dados['intervaloFundacao'],
        novo_cliente_dados['capitalSocial'],
        novo_cliente_dados['restricoes'],
        novo_cliente_dados['empresa_MeEppMei'],
        novo_cliente_dados['scorePontualidade'],
        novo_cliente_dados['limiteEmpresaAnaliseCredito'],
        novo_cliente_dados['primeiraCompra'],  # Exibido diretamente como string
        int(novo_cliente_dados['anoFundacao'])  # Convertido para inteiro para melhor exibição
    ]

    # Criar labels para dados
    for i in range(len(labels)):
        lbl = ttk.Label(frame_dados, text=f"{labels[i]} {valores[i]}", wraplength=550, justify='left')
        lbl.pack(anchor='w', padx=10, pady=2)

    # Frame para a recomendação
    frame_prev = ttk.LabelFrame(janela, text="Limite de Crédito Recomendado")
    frame_prev.pack(padx=10, pady=10, fill="both", expand=True)

    # Exibir a recomendação fixa
    lbl_limite = ttk.Label(frame_prev, text=f"Limite de Crédito: R$ {limite_novo_cliente:,.2f}", font=("Arial", 12))
    lbl_limite.pack(pady=5)

    lbl_categoria = ttk.Label(frame_prev, text=f"Categoria: {categoria_limite}", font=("Arial", 12))
    lbl_categoria.pack(pady=5)

    # Botão para sair
    btn_sair = ttk.Button(janela, text="Sair", command=janela.destroy)
    btn_sair.pack(pady=10)

    # Executar a janela
    janela.mainloop()

# Chamar a função para criar a GUI
criar_gui()
