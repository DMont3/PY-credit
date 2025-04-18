
## Descrição do Projeto

Este projeto tem como objetivo implementar e testar modelos de machine learning para recomendação de crédito e segmentação de clientes. Ele inclui uma série de scripts Python que realizam a carga, processamento, segmentação e modelagem de dados de clientes, utilizando diversas técnicas de aprendizado de máquina. O projeto também inclui visualizações para analisar os resultados gerados, como gráficos e modelos treinados.

## Estrutura do Projeto

A estrutura de diretórios do projeto é a seguinte:

- **src/**: Contém os scripts Python principais para o processamento de dados e modelagem.
- **data/**: Diretório onde todos os dados são armazenados.
- **figures/**: Diretório para salvar gráficos gerados durante a execução dos scripts.
- **venv/**: Ambiente virtual Python usado no projeto.
- **.gitignore**: Arquivo para especificar quais arquivos ou pastas devem ser ignorados pelo Git.
- **requirements.txt**: Arquivo que lista todas as dependências do projeto.
- **README.md**: Arquivo de documentação do projeto.

## Funcionalidades

1. **Segmentação de Clientes**: Scripts para segmentar clientes com base em características específicas, usando técnicas de clustering.
2. **Recomendação de Crédito**: Modelos de machine learning que recomendam limites de crédito com base em dados históricos.
3. **Visualização de Resultados**: Gráficos e outras saídas visuais geradas durante o processo de análise.

## Pré-requisitos

- Python 3.10 ou superior
- Virtualenv (opcional, mas recomendado)


## Uso

1. Coloque seus dados no diretório `data/`.
2. Para processar os dados e gerar os resultados, execute o script principal:

    ```bash
    python src/avaliacao_credito.py
    ```

