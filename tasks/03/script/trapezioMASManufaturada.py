# autor: Igor Pontes Tresolavy
# autor: Thiago Antici Rodrigues de Souza

"""
Confeccionando tabelas de convergência para o problema manufaturado:
y_e(t) = e^(-t) - e^(-4t)

Problema de Cauchy 2D
d[(y1(t), y2(t))]/dt = ((0, 1), (-4, -5))*(y1(t), y2(t))
(y1(0), y2(0)) = (0, 3)

Há 9 parâmetros nesse programa:
    NO_DE_CASOS: define a quantidade de linhas na tabela da saída. Ou seja, define quantas malhas progressivamente mais finas que o programa executa.
    INICIO_INTERVALO: define o início do intervalo o qual se deseja discretizar através o Método do Trapézio Implícito

Adicionalmente, pode-se alterar a função f do Problema de Cauchy e a solução conhecida a fim de reutilizar o programa para a discretização de outro problema
"""

import matplotlib.pyplot as plt
import numpy as np

# parâmetros
NO_DE_CASOS = 10
INICIO_INTERVALO = 0
CONDICAO_INICIAL = [0, 3]
FIM_INTERVALO = (3.0*np.pi)/4.0
QNTD_PASSOS_INICIAL = 8
FATOR_MULTIPLICATIVO = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL
TOL = 2e-8
MAXIMO_ITERACOES = 3

# Solução exata do Problema de Cauchy 2D
def y_e(T):
    return np.array([np.exp(-T) - np.exp(-4*T), -np.exp(-T) + 4*np.exp(-4*T)])

# f(t,y_1, y_2) do problema de Cauchy 2D
def f(t, y):
    return np.array([y[1], -4*y[0] -5*y[1]])

# Método do Trapézio Implícito com Iterações de Newton
def trapezoidal(t_0, T, h_n, f, w_0):

    t = np.arange(t_0, T + h_n, h_n)

    # condição inicial
    y = [np.array(w_0)]

    for t_k in t[:-1]:
        k_1 = y[-1] + (h_n*f(t_k,y[-1]))/2
        # o chute inicial é resultado da aplicação do Método de Euler
        y_j = y[-1] + h_n*f(t_k,y[-1])

        # Método das aproximações sucessivas
        j = 1
        flag = 0
        while flag==0:
            w = (k_1 + h_n*f(t_k + h_n, y_j)/2)
            if abs(np.max((w - y_j))/np.max(y_j)) < TOL:
                flag = 1
            else:
                j = j + 1
                y_j = w
                if j > MAXIMO_ITERACOES:
                   break

        y.append(w)

    return t, y

def main():

    # Gráficos para as variáveis de estado
    fig, (ax1, ax2) = plt.subplots(2, figsize=(7,10))
    fig.tight_layout()
    linestyles = ["dotted", "dashed", "dashdot", (0, (10, 3))]

    print("TABELA DE VERIFICAÇÃO DE SOLUÇÃO MANUFATURADA");

    for caso in range(1, NO_DE_CASOS + 1):
        # quantidade de subdivisões no intervalos
        n = QNTD_PASSOS_INICIAL*(FATOR_MULTIPLICATIVO**(caso-1))

        # tamanho do passo de integração
        h_n = (FIM_INTERVALO - INICIO_INTERVALO)/n

        # aproximação da solução no intervalo e pontos de malha do tempo
        malha__do__tempo, aproximacao__numerica = trapezoidal(INICIO_INTERVALO, FIM_INTERVALO, h_n, f, CONDICAO_INICIAL)

        # norma euclidiana do erro de discretização global
        e = np.linalg.norm(y_e(FIM_INTERVALO) - aproximacao__numerica[-1])

        # aproximação da ordem do método
        ordem__p = np.log(abs(e__anterior/e))/np.log(FATOR_MULTIPLICATIVO) if caso != 1 else 0

        print("%s %5d %s & %s %9.3e %s & %s %9.3e %s & %s %9.3e %s  \\\\" % ("$", n, "$", "$", h_n, "$", "$",e, "$", "$",ordem__p, "$"));

        e__anterior = e

        # traçado de gráficos para determinados casos
        if caso == 2 or caso == 3 or caso == 8:
            ax1.plot(malha__do__tempo, [y[0] for y in aproximacao__numerica], color="black", linestyle=linestyles[caso%4], label="$\eta_1(t,h)$ para n = {}".format(str(n)))
            ax2.plot(malha__do__tempo, [y[1] for y in aproximacao__numerica], color="black", linestyle=linestyles[caso%4], label="$\eta_2(t,h)$ para n = {}".format(str(n)))



    ax1.plot(malha__do__tempo, y_e(malha__do__tempo)[0], color="black", linestyle="solid", label="solução exata")
    ax2.plot(malha__do__tempo, y_e(malha__do__tempo)[1], color="black", linestyle="solid", label="primeira derivada da solução exata")

    ax1.set_xlabel("t (adimensional)")
    ax1.set_ylabel("variável de estado (adimensional)")
    ax1.set_title("Método de Euler 2D para a variável de estado $y_1$ \n(solução exata $y_e(t) = e^{-t} - e^{-4t}$)", size=12)
    ax1.legend(loc="upper right",)

    ax2.set_xlabel("t (adimensional)")
    ax2.set_ylabel("variável de estado (adimensional)")
    ax2.set_title("Método de Euler 2D para a variável de estado $y_2$ \n(derivada da solução exata $\dot{y}_e(t) = -e^{-t} + 4e^{-4t}$)", size=12)
    ax2.legend(loc="upper right")

    ax1.set_box_aspect(1/2)
    ax2.set_box_aspect(1/2)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
