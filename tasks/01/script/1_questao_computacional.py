# autor: Igor Pontes Tresolavy
# autor: Thiago Antice Rodrigues de Souza

"""
Traçando o gráfico y(t) = cos(mt), -pi <= t <= pi
para m inteiro variando de 1 a 3
"""

import matplotlib.pyplot as plt
import numpy as np

# definições
INICIO_INTERVALO = -np.pi
FIM_INTERVALO = np.pi
QNTD_PASSOS = 10000

def main():

    # definindo o passo no
    # intervalo de -pi a pi
    t_0 = INICIO_INTERVALO
    T = FIM_INTERVALO
    delta_t = (T - t_0)/QNTD_PASSOS

    # definindo a sequência de passos no tempo
    t = np.arange(t_0, T + delta_t, delta_t)

    # definindo funções a serem traçadas
    y_1 = np.cos(t)
    y_2 = np.cos(2*t)
    y_3 = np.cos(3*t)

    fig, ax = plt.subplots(figsize=(3*np.pi,3))

    # traçando o gráfico das funções
    cos1 = ax.plot(t, y_1, color="black", linestyle="dotted",
            label="cos(t) adimensional")
    cos2 = ax.plot(t, y_2, color="black", linestyle="dashed",
            label="cos(2t) adimensional")
    cos3 = ax.plot(t, y_3, color="black", linestyle="dashdot",
            label="cos(3t) adimensional")

    # adicionando descrições ao gráfico
    ax.set_xlabel("ângulo t (em rad)")
    ax.set_ylabel("variáveis de estado")
    ax.set_title("Traçado das funções coseno para 3 frequências angulares distintas", size=12)
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left",)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
