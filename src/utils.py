import os
import matplotlib.pyplot as plt


def salvar_figura(fig_id, diretorio="..\\figures", extensao_figura="png", resolucao=300):
    # Criar o diretório se ele não existir
    caminho = os.path.join(diretorio, fig_id)
    os.makedirs(os.path.dirname(caminho), exist_ok=True)

    # Salvar a figura
    plt.savefig(caminho, format=extensao_figura, dpi=resolucao)
    print(f"Figura salva em {caminho}")
