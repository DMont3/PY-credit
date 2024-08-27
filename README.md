# CP04-Python

## Descrição do Projeto

Este projeto tem como objetivo implementar e testar modelos de machine learning para recomendação de crédito e segmentação de clientes. Ele inclui uma série de scripts Python que realizam a carga, processamento, segmentação e modelagem de dados de clientes, utilizando diversas técnicas de aprendizado de máquina. O projeto também inclui uma interface para salvar e visualizar os resultados gerados, como gráficos e modelos treinados.

## Estrutura do Projeto

A estrutura de diretórios do projeto é a seguinte:


- **src/**: Contém os scripts Python principais para o processamento de dados e modelagem.
- **data/raw/**: Diretório onde os dados brutos são armazenados.
- **data/processed/**: Diretório para dados processados e segmentados.
- **figures/**: Diretório para salvar gráficos gerados durante a execução dos scripts.
- **venv/**: Ambiente virtual Python usado no projeto.

## Funcionalidades

1. **Segmentação de Clientes**: Scripts para segmentar clientes com base em características específicas, usando clustering.
2. **Recomendação de Crédito**: Modelos de machine learning que recomendam limites de crédito com base em dados históricos.
3. **Visualização de Resultados**: Gráficos e outras saídas visuais geradas durante o processo de análise.

## Pré-requisitos

- Python 3.10 ou superior
- Virtualenv (opcional, mas recomendado)

## Instalação

1. Clone este repositório:

   ````bash
   1. Clone este repositório:
   
   git clone https://github.com/DMont3/CP04-Python
   
   2. Navegue até o diretório do projeto:

   cd CP04-Python

   3. Crie e ative um ambiente virtual:

   python -m venv venv
   source venv/bin/activate  # No Windows use venv\Scripts\activate

   4. Instale as dependências:

   pip install -r requirements.txt

## Uso

1. Coloque seus dados brutos no diretório `data/raw/`.
2. Para processar os dados e gerar os resultados, rode o pipeline com o script `run_pipeline.py`:

   ````bash
   python src/run_pipeline.py
