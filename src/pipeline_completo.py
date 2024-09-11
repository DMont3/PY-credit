import os
import logging
import subprocess
from datetime import datetime
import sys
import importlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def criar_diretorio_saida():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diretorio_saida = f"resultados_{timestamp}"
    os.makedirs(diretorio_saida, exist_ok=True)
    return diretorio_saida


def check_dependencies():
    required_modules = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'joblib']
    missing_modules = []

    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        logging.error(f"The following modules are missing: {', '.join(missing_modules)}")
        logging.error("Please install them using: pip install " + " ".join(missing_modules))
        return False
    return True


def executar_script(nome_script, diretorio_saida, **kwargs):
    logging.info(f"Executando {nome_script}...")
    comando = [sys.executable, nome_script]

    for key, value in kwargs.items():
        comando.extend([f"--{key}", str(value)])

    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True)

        # Salvar logs
        with open(os.path.join(diretorio_saida, f"{nome_script}_log.txt"), "w") as log_file:
            log_file.write(resultado.stdout)
            if resultado.stderr:
                log_file.write("\n--- Erros ---\n")
                log_file.write(resultado.stderr)

        logging.info(f"{nome_script} executado com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar {nome_script}")
        logging.error(e.stderr)
        return False
    except Exception as e:
        logging.error(f"Erro inesperado ao executar {nome_script}: {str(e)}")
        return False


def main():
    if not check_dependencies():
        return

    diretorio_saida = criar_diretorio_saida()
    logging.info(f"Resultados serão salvos em: {diretorio_saida}")

    caminho_dados = input("Insira o caminho para o arquivo de dados CSV: ")

    etapas = [
        # Step 1: Preprocessing
        ("preprocessamento.py", {
            "input": caminho_dados,
            "output": os.path.join(diretorio_saida, "dados_preprocessados.csv")
        }),

        # Step 2: Client Grouping
        ("client_grouping.py", {
            "input": os.path.join(diretorio_saida, "dados_preprocessados.csv"),
            "output": os.path.join(diretorio_saida, "group_classifier.joblib")
        }),

        # Step 3: Model Training
        ("model_training.py", {
            "input": os.path.join(diretorio_saida, "dados_preprocessados.csv"),
            "output": os.path.join(diretorio_saida, "trained_model.joblib")
        }),

        # Step 4: Make Predictions
        ("fazer_predicoes.py", {
            "input": os.path.join(diretorio_saida, "dados_preprocessados.csv"),
            "modelo": os.path.join(diretorio_saida, "trained_model.joblib"),
            "output": os.path.join(diretorio_saida, "predicoes.csv")
        }),

        # Step 5: Visualization
        ("visualizacao_resultados.py", {
            "input": os.path.join(diretorio_saida, "predicoes.csv"),
            "output": os.path.join(diretorio_saida, "resultados_visualizados.png")
        })
    ]

    for script, args in etapas:
        if not executar_script(script, diretorio_saida, **args):
            logging.error(f"Pipeline interrompido devido a erro em {script}")
            return

    logging.info("Pipeline completo executado com sucesso")
    logging.info(f"Todos os resultados foram salvos em: {diretorio_saida}")


if __name__ == "__main__":
    main()