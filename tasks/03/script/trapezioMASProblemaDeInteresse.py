# autor: Igor Pontes Tresolavy
# autor: Thiago Antici Rodrigues de Souza

"""
Esse programa confecciona tabelas de convergência para o seguinte
problema de Lotka-Volterra (Lotka-Volterra com características de relacionamento mutual;
ver https://en.wikipedia.org/wiki/Mutualism_(biology)#Mathematical_modeling):

dp/dt = a*p(1-p/K) - b*p*q/(1+b*p)
dq/dt = m*q(1-1/(k*p))

com a = 0.2, m = 0.1, K= 500, k = 0.2, b = 0.1, p(0) = 10,
q(0) = 5

Há 9 parâmetros nesse programa:

    NO_DE_CASOS: define a quantidade de linhas na tabela da saída.
                 Ou seja, define quantas malhas progressivamente
                 mais finas que o programa executa.

    INICIO_INTERVALO: define o início do intervalo o qual se deseja
                      discretizar a solução da EDO através o Método
                      do Trapézio Implícito

    CONDICAO_INICIAL: condição inicial do Problema de Cauchy

    FIM_INTERVALO: define o fim do intervalo o qual se deseja
                   discretizar a solução da EDO através o Método
                   do Trapézio Implícito

    QNTD_PASSOS_INICIAL: define a primeira quantidade de subintervalos
                         no intervalo [INICIO_INTERVALO, FIM_INTERVALO]

    FATOR_MULTIPLICATIVO: define em quantas vezes a quantidade de
                          subintervalos é diminuída a cada caso

    TOL: define a tolerância do erro relativo no critério de para do
         Método do Ponto Fixo

    MAXIMO_ITERACOES: define a quantidade máxima de iterações do
                      Método do Ponto Fixo

Adicionalmente, pode-se alterar a função f do Problema de Cauchy a fim
de se reutilizar o programa para a discretização de outro problema pelo
Método Trapézio Implícito.
"""

import matplotlib.pyplot as plt
import numpy as np

# definições
NO_DE_CASOS = 8
INICIO_INTERVALO = 0
CONDICAO_INICIAL = [10, 5]
FIM_INTERVALO = 80
QNTD_PASSOS_INICIAL = 32
FATOR_MULTIPLICATIVO = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL
TOL = 2e-8
MAXIMO_ITERACOES = 3

# Método do Trapézio Implícito com Iterações de Newton
def trapezoidal(t_0, T, h_n, f, w_0):

    t = np.arange(t_0, T + h_n, h_n)
    y = [np.array(w_0)]

    for t_k in t[:-1]:
        k_1 = y[-1] + (h_n*f(t_k,y[-1]))/2
        y_m = y[-1] + h_n*f(t_k,y[-1])

        # Método das aproximações sucessivas
        m = 1
        flag = 0
        while flag==0:
            w = (k_1 + h_n*f(t_k + h_n, y_m)/2)
            if abs(np.linalg.norm((w - y_m))/np.linalg.norm(y_m)) < TOL:
                flag = 1
            else:
                m = m + 1
                y_m = w
                if m > MAXIMO_ITERACOES:
                   break

        y.append(w)

    return t, y

# f(t,y_1, y_2) do problema de Cauchy 2D
def f(t, y):
    p = y[0]
    q = y[1]
    a = 0.2
    m = 0.1
    K = 500
    k = 0.2
    b = 0.1
    return np.array([a*p*(1-p/K) - b*p*q/(1+b*p), m*q*(1-1/(k*p))])


def main():

    fig, (ax1, ax2) = plt.subplots(2, figsize=(7,10))
    fig.tight_layout()
    linestyles = ["dotted", "dashed", "dashdot", (0, (10, 3))]

    print("TABELA DE VERIFICAÇÃO DE SOLUÇÃO MANUFATURADA (1D e 2D)");

    for caso in range(1, NO_DE_CASOS + 1):
        n = QNTD_PASSOS_INICIAL*(FATOR_MULTIPLICATIVO**(caso-1))
        h_n = (FIM_INTERVALO - INICIO_INTERVALO)/n

        malha__do__tempo, aproximacao__numerica = trapezoidal(INICIO_INTERVALO, FIM_INTERVALO, h_n, f, CONDICAO_INICIAL)
        proxima__malha__do__tempo, proxima__aproximacao__numerica = trapezoidal(INICIO_INTERVALO, FIM_INTERVALO, h_n/2, f, CONDICAO_INICIAL)

        # norma euclidiana do erro de discretização global
        e = np.linalg.norm(aproximacao__numerica[-1] - proxima__aproximacao__numerica[-1])

        ordem__p = np.log(abs(e__anterior/e))/np.log(FATOR_MULTIPLICATIVO) if caso != 1 else 0

        print("%s %5d %s & %s %9.3e %s & %s %9.3e %s & %s %9.3e %s  \\\\" % ("$", n, "$", "$", h_n, "$", "$",e, "$", "$",ordem__p, "$"));

        e__anterior = e

        if caso == 1 or caso == 2 or caso == 3 or caso == 8:
            ax1.plot(malha__do__tempo, [y[0] for y in aproximacao__numerica], color="black", linestyle=linestyles[caso%4], label="$n = {}".format(str(n)))
            ax2.plot(malha__do__tempo, [y[1] for y in aproximacao__numerica], color="black", linestyle=linestyles[caso%4], label="$n = {}".format(str(n)))

    ax1.set_xlabel("t (adimensional)")
    ax1.set_ylabel("variável de estado (adimensional)")
    ax1.set_title("Aproximações de $p(t)$ para diversos valores de $n$", size=12)
    ax1.legend(loc="upper left")

    ax2.set_xlabel("t (adimensional)")
    ax2.set_ylabel("variável de estado (adimensional)")
    ax2.set_title("Aproximações de $q(t)$ para diversos valores de $n$", size=12)
    ax2.legend(loc="upper left")

    ax1.set_box_aspect(1/2)
    ax2.set_box_aspect(1/2)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
