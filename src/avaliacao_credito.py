import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

sns.set(style='whitegrid')

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
figures_dir = os.path.join(script_dir, '..', 'figures')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

input_csv_path = os.path.join(data_dir, 'sample_data.csv')

df = pd.read_csv(input_csv_path, encoding='utf-8')

print("primeiras linhas do dataset:")
print(df.head())

print("\ninformações do dataset:")
print(df.info())

dados = df.copy()

colunas_remover = [
    'numero_solicitacao', 'razaoSocial', 'nomeFantasia', 'cnpjSemTraco',
    'dataAprovadoEmComite', 'dashboardCorrelacao', 'diferencaPercentualRisco'
]
dados = dados.drop(columns=colunas_remover, errors='ignore')

dados = dados.dropna(subset=['valorAprovado'])  #remove rows where 'valorAprovado' is missing

colunas_data = ['primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite', 'periodoBalanco']
for col in colunas_data:
    if col in dados.columns:
        dados[col] = pd.to_datetime(dados[col], errors='coerce')

ano_atual = datetime.today().year
dados['idade_empresa'] = ano_atual - dados['anoFundacao']
dados['dias_desde_primeira_compra'] = (datetime.today() - dados['primeiraCompra']).dt.days

colunas_categoricas = ['status', 'definicaoRisco', 'intervaloFundacao']
label_encoders = {}
for col in colunas_categoricas:
    if col in dados.columns:
        le = LabelEncoder()
        dados[col] = le.fit_transform(dados[col].astype(str))
        label_encoders[col] = le  #save labelencoder for later

colunas_booleanas = ['restricoes', 'empresa_MeEppMei']
for col in colunas_booleanas:
    if col in dados.columns:
        dados[col] = dados[col].astype(str).map({'False': 0, 'True': 1, 'nan': 0})

colunas_remover = [
    'primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite',
    'periodoBalanco', 'anoFundacao'
]
dados = dados.drop(columns=colunas_remover, errors='ignore')

imputer = SimpleImputer(strategy='median')
dados = pd.DataFrame(imputer.fit_transform(dados), columns=dados.columns)

print("\ntipos de dados antes do escalonamento:")
print(dados.dtypes)

colunas_nao_numericas = dados.select_dtypes(include=['object']).columns
print("\ncolunas não numéricas:")
print(colunas_nao_numericas)

dados = dados.drop(columns=colunas_nao_numericas, errors='ignore')

scaler = StandardScaler()
features = dados.drop('valorAprovado', axis=1).columns
dados[features] = scaler.fit_transform(dados[features])

print("\ndataframe processado:")
print(dados.head())

#segmentação de clientes

x_cluster = dados.drop('valorAprovado', axis=1)

inercia = []
k = range(2, 10)
for k in k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_cluster)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k, inercia, 'bx-')
plt.xlabel('número de clusters (k)')
plt.ylabel('inércia')
plt.title('método do cotovelo para determinar k ótimo')
cotovelo_path = os.path.join(figures_dir, 'metodo_cotovelo.png')
plt.savefig(cotovelo_path)
plt.show()

k_otimo = 5
kmeans = KMeans(n_clusters=k_otimo, random_state=42)
dados['Grupo'] = kmeans.fit_predict(x_cluster)

pca = PCA(n_components=2)
componentes_principais = pca.fit_transform(x_cluster)
pc_df = pd.DataFrame(data=componentes_principais, columns=['PCA 1', 'PCA 2'])
pc_df['Grupo'] = dados['Grupo']

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA 1', y='PCA 2', hue='Grupo', data=pc_df, palette='Set1')
plt.title('segmentação de clientes (pca)')
segmentacao_path = os.path.join(figures_dir, 'segmentacao_clientes.png')
plt.savefig(segmentacao_path)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Grupo', data=dados, palette='Set2')
plt.xlabel('grupo de clientes')
plt.ylabel('quantidade de clientes')
plt.title('distribuição de clientes por grupo')
distribuicao_grupos_path = os.path.join(figures_dir, 'distribuicao_grupos.png')
plt.savefig(distribuicao_grupos_path)
plt.show()

resumo_grupos = dados.groupby('Grupo').mean()
print("\nresumo dos grupos:")
print(resumo_grupos)

df_grupos = df.copy()
df_grupos['Grupo'] = dados['Grupo']
clientes_segmentados_path = os.path.join(data_dir, 'clientes_segmentados.csv')
df_grupos.to_csv(clientes_segmentados_path, index=False)  #save to data folder

#classificação de novos clientes

x = dados.drop(['Grupo', 'valorAprovado'], axis=1)
y = dados['Grupo']

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.2, random_state=42
)

features_numericas = x.select_dtypes(include=['float64', 'int64']).columns
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

pipeline_clf.fit(x_treino, y_treino)

y_pred = pipeline_clf.predict(x_teste)
print("\nrelatório de classificação:")
print(classification_report(y_teste, y_pred))

matriz_confusao = confusion_matrix(y_teste, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('previsto')
plt.ylabel('real')
plt.title('matriz de confusão')
matriz_confusao_path = os.path.join(figures_dir, 'matriz_confusao.png')
plt.savefig(matriz_confusao_path)
plt.show()

#gráfico de importância das features
modelo_rf = pipeline_clf.named_steps['classifier']

importancias = modelo_rf.feature_importances_
nomes_features = x.columns

df_importancias = pd.DataFrame({
    'Feature': nomes_features,
    'Importancia': importancias
})

df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importancia', y='Feature', data=df_importancias, palette='viridis')
plt.xlabel('importância da feature')
plt.ylabel('feature')
plt.title('importância das features no modelo de classificação')
plt.tight_layout()
importancia_features_path = os.path.join(figures_dir, 'importancia_features.png')
plt.savefig(importancia_features_path)
plt.show()

#recomendação automatizada de crédito

def categorizar_limite_credito(valor):
    if valor <= 10000:
        return 'Baixo'
    elif valor <= 50000:
        return 'Médio'
    elif valor <= 100000:
        return 'Alto'
    else:
        return 'Muito Alto'

dados['Categoria_Limite_Credito'] = dados['valorAprovado'].apply(categorizar_limite_credito)

print("colunas do dataframe 'dados' após criar 'categoria_limite_credito':")
print(dados.columns)

plt.figure(figsize=(8, 6))
categorias_ordem = ['Baixo', 'Médio', 'Alto', 'Muito Alto']
sns.countplot(x='Categoria_Limite_Credito', data=dados, palette='Set2', order=categorias_ordem)
plt.xlabel('categoria do limite de crédito')
plt.ylabel('quantidade de clientes')
plt.title('distribuição das categorias de limite de crédito')
distribuicao_categorias_limite_path = os.path.join(figures_dir, 'distribuicao_categorias_limite.png')
plt.savefig(distribuicao_categorias_limite_path)
plt.show()

dados_reg = dados.copy()

x_reg = dados_reg.drop(['valorAprovado', 'Grupo', 'Categoria_Limite_Credito'], axis=1)
y_reg = dados_reg['valorAprovado']

x_treino_reg, x_teste_reg, y_treino_reg, y_teste_reg = train_test_split(
    x_reg, y_reg, test_size=0.2, random_state=42
)

features_numericas_reg = x_reg.select_dtypes(include=['float64', 'int64']).columns
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

pipeline_reg.fit(x_treino_reg, y_treino_reg)

y_pred_reg = pipeline_reg.predict(x_teste_reg)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_teste_reg, y_pred_reg)
r2 = r2_score(y_teste_reg, y_pred_reg)
print(f"\nerro médio quadrático (mse): {mse}")
print(f"coeficiente de determinação (r²): {r2}")

plt.figure(figsize=(8, 6))
plt.scatter(y_teste_reg, y_pred_reg, alpha=0.5, color='purple')
plt.plot(
    [y_teste_reg.min(), y_teste_reg.max()],
    [y_teste_reg.min(), y_teste_reg.max()],
    'r--'
)
plt.xlabel('valor aprovado real (r$)')
plt.ylabel('valor aprovado previsto (r$)')
plt.title('comparação entre valores reais e previstos')
regressao_credito_path = os.path.join(figures_dir, 'regressao_credito.png')
plt.savefig(regressao_credito_path)
plt.show()

def classificar_novo_cliente(novo_cliente):
    novo_cliente_df = pd.DataFrame([novo_cliente])

    for col in colunas_categoricas:
        if col in novo_cliente_df.columns:
            le = label_encoders[col]
            novo_cliente_df[col] = le.transform(novo_cliente_df[col].astype(str))

    for col in colunas_booleanas:
        if col in novo_cliente_df.columns:
            novo_cliente_df[col] = novo_cliente_df[col].astype(str).map({'False': 0, 'True': 1, 'nan': 0})

    novo_cliente_df['idade_empresa'] = ano_atual - novo_cliente_df['anoFundacao']
    novo_cliente_df['dias_desde_primeira_compra'] = (datetime.today() - pd.to_datetime(novo_cliente_df['primeiraCompra'])).dt.days

    colunas_remover = ['primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite', 'periodoBalanco', 'anoFundacao']
    novo_cliente_df = novo_cliente_df.drop(columns=colunas_remover, errors='ignore')

    novo_cliente_df = novo_cliente_df[x_treino.columns]

    grupo_label = pipeline_clf.predict(novo_cliente_df)
    return grupo_label[0]

def recomendar_limite_credito(dados_cliente):
    cliente_df = pd.DataFrame([dados_cliente])

    for col in colunas_categoricas:
        if col in cliente_df.columns:
            le = label_encoders[col]
            cliente_df[col] = le.transform(cliente_df[col].astype(str))

    for col in colunas_booleanas:
        if col in cliente_df.columns:
            cliente_df[col] = cliente_df[col].astype(str).map({'False': 0, 'True': 1, 'nan': 0})

    cliente_df['idade_empresa'] = ano_atual - cliente_df['anoFundacao']
    cliente_df['dias_desde_primeira_compra'] = (datetime.today() - pd.to_datetime(cliente_df['primeiraCompra'])).dt.days

    colunas_remover = ['primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite', 'periodoBalanco', 'anoFundacao']
    cliente_df = cliente_df.drop(columns=colunas_remover, errors='ignore')

    cliente_df = cliente_df[x_reg.columns]

    limite_previsto = pipeline_reg.predict(cliente_df)[0]

    limite_previsto = max(0, min(limite_previsto, 150000))

    return limite_previsto

df_recomendacoes = df.copy()
df_recomendacoes['Grupo'] = dados['Grupo']
df_recomendacoes['Limite_Recomendado'] = dados['valorAprovado']
df_recomendacoes['Categoria_Limite_Credito'] = dados['Categoria_Limite_Credito']
clientes_recomendacoes_path = os.path.join(data_dir, 'clientes_recomendacoes.csv')
df_recomendacoes.to_csv(clientes_recomendacoes_path, index=False)

plt.figure(figsize=(12, 6))

limite_max = 150000

dados_filtrados = dados[dados['valorAprovado'] <= limite_max]

sns.histplot(dados_filtrados['valorAprovado'], bins='auto', kde=False, color='blue')

plt.xlabel('valor aprovado (r$)')
plt.ylabel('frequência')
plt.title('distribuição dos limites de crédito aprovados (valores até r$150.000)')

plt.xticks(rotation=45)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

distribuicao_limites_path = os.path.join(figures_dir, 'distribuicao_limites.png')
plt.savefig(distribuicao_limites_path)
plt.show()

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

grupo_novo_cliente = classificar_novo_cliente(novo_cliente_dados)
print(f"o novo cliente pertence ao grupo: {grupo_novo_cliente}")

limite_novo_cliente = recomendar_limite_credito(novo_cliente_dados)
categoria_limite = categorizar_limite_credito(limite_novo_cliente)
print(f"limite de crédito recomendado: r$ {limite_novo_cliente:,.2f} - categoria: {categoria_limite}")

def criar_gui():
    janela = tk.Tk()
    janela.title("Previsão de Limite de Crédito")
    janela.geometry("600x1000")
    janela.resizable(False, False)

    titulo = ttk.Label(janela, text="Recomendação de Limite de Crédito", font=("Arial", 16))
    titulo.pack(pady=10)

    frame_dados = ttk.LabelFrame(janela, text="Dados do Cliente")
    frame_dados.pack(padx=10, pady=10, fill="both", expand=True)

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
        novo_cliente_dados['primeiraCompra'],
        int(novo_cliente_dados['anoFundacao'])
    ]

    for i in range(len(labels)):
        lbl = ttk.Label(frame_dados, text=f"{labels[i]} {valores[i]}", wraplength=550, justify='left')
        lbl.pack(anchor='w', padx=10, pady=2)

    frame_prev = ttk.LabelFrame(janela, text="Limite de Crédito Recomendado")
    frame_prev.pack(padx=10, pady=10, fill="both", expand=True)

    lbl_limite = ttk.Label(frame_prev, text=f"Limite de Crédito: R$ {limite_novo_cliente:,.2f}", font=("Arial", 12))
    lbl_limite.pack(pady=5)

    lbl_categoria = ttk.Label(frame_prev, text=f"Categoria: {categoria_limite}", font=("Arial", 12))
    lbl_categoria.pack(pady=5)

    btn_sair = ttk.Button(janela, text="Sair", command=janela.destroy)
    btn_sair.pack(pady=10)

    janela.mainloop()

criar_gui()
