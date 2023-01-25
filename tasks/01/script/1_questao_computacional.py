# autor: Igor Pontes Tresolavy
# autor: Thiago Antici Rodrigues de Souza

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

    # traçando o gráfico das funções
    plt.plot(t, y_1, color="black", linestyle="dotted",
            label="cos(t) adimensional")
    plt.plot(t, y_2, color="black", linestyle="dashed",
            label="cos(2t) adimensional")
    plt.plot(t, y_3, color="black", linestyle="dashdot",
            label="cos(3t) adimensional")

    # adicionando descrições ao gráfico
    plt.xlabel("ângulo t (em rad)")
    plt.ylabel("variáveis de estado")
    plt.title("Traçado das funções coseno para 3 frequências angulares distintas")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
