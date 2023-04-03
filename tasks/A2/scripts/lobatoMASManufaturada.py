# autor: Igor Pontes Tresolavy
# autor: Thiago Antici Rodrigues de Souza

"""
Confeccionando tabelas de convergência para o problema manufaturado:
y_e(t) = (y_1(t), y_2(t)) =
(
e^(-t)*cos(t),
6e^(-t)sin(t)
)

Problema de Cauchy 2D
d[(y1(t), y2(t))]/dt = ((-1, -1/6), (6, -1))*(y1(t), y2(t))
(y_1(0), y_2(0)) = (1, 0)

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

Adicionalmente, pode-se alterar a função f do Problema de Cauchy e a
solução conhecida a fim de reutilizar o programa para a discretização
de outro problema
"""

import matplotlib.pyplot as plt
import numpy as np

# parâmetros
NO_DE_CASOS = 10
INICIO_INTERVALO = 0
CONDICAO_INICIAL = [1, 0]
FIM_INTERVALO = 100
QNTD_PASSOS_INICIAL = 4
FATOR_MULTIPLICATIVO = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL
TOL = 2e-8
MAXIMO_ITERACOES = 3

# Solução exata do Problema de Cauchy 2D
def y_e(T):
    return np.array([np.exp(-T)*np.cos(T), 6*np.exp(-T)*np.sin(T)])

# f(t,y_1, y_2) do problema de Cauchy 2D
def f(t, y):
    return np.array([-1*y[0] -y[1]/6, 6*y[0] -1*y[1]])

# Método LobatoIIIC de segunda ordem
# com Método das Aproximações Sucessivas
def _lobatoIIIC_ordem2(t_0, T, h_n, f, y_0):

    t = np.arange(t_0, T + h_n, h_n)

    # condição inicial
    y = [np.array(y_0)]

    for t_k in t[:-1]:
        # o chute inicial dos coeficientes é resultado
        # da aplicação método de Euler Modificado
        k_1 = f(t_k      , y[-1])
        k_2 = f(t_k + h_n, y[-1] + h_n*k_1)

        # Método das aproximações sucessivas
        j     = 1
        k_1_j = k_1
        k_2_j = k_2
        flag  = 0
        while flag == 0:

            k_1 = f(t_k      , y[-1] + h_n*(k_1_j - k_2_j)/2)
            k_2 = f(t_k + h_n, y[-1] + h_n*(k_1_j + k_2_j)/2)

            if max(abs(np.linalg.norm((k_1_j - k_1))/np.linalg.norm(k_1)), abs(np.linalg.norm((k_2_j - k_2))/np.linalg.norm(k_2))) < TOL:
                flag = 1
            else:
                j     = j + 1
                k_1_j = k_1
                k_2_j = k_2
                if j > MAXIMO_ITERACOES:
                   break

        y.append(y[-1] + h_n*((1/2)*k_1 + (1/2)*k_2))

    return t, y

# Método LobatoIIIC de quarta ordem
# com Método das Aproximações Sucessivas
def _lobatoIIIC_ordem4(t_0, T, h_n, f, y_0):

    t = np.arange(t_0, T + h_n, h_n)

    # condição inicial
    y = [np.array(y_0)]

    for t_k in t[:-1]:
        # o chute inicial dos coeficientes é resultado
        # da aplicação do Runge-Kutta de ordem 3 (RK33)
        k_1 = f(t_k        , y[-1])
        k_2 = f(t_k + h_n/2, y[-1] + h_n*k_1/2)
        k_3 = f(t_k + h_n  , y[-1] + h_n*(-k_1 + 2*k_2))

        # Método das aproximações sucessivas
        j     = 1
        k_1_j = k_1
        k_2_j = k_2
        k_3_j = k_3
        flag  = 0
        while flag == 0:

            k_1 = f(t_k        , y[-1] + h_n*(k_1_j/6 -   k_2_j/3  + k_3_j/6))
            k_2 = f(t_k + h_n/2, y[-1] + h_n*(k_1_j/6 + 5*k_2_j/12 - k_3_j/12))
            k_3 = f(t_k + h_n  , y[-1] + h_n*(k_1_j/6 + 2*k_2_j/3  + k_3_j/6))

            if max(abs(np.linalg.norm((k_1_j - k_1))/np.linalg.norm(k_1)),
                   abs(np.linalg.norm((k_2_j - k_2))/np.linalg.norm(k_2)),
                   abs(np.linalg.norm((k_3_j - k_3))/np.linalg.norm(k_3))) < TOL:
                flag = 1
            else:
                j     = j + 1
                k_1_j = k_1
                k_2_j = k_2
                k_3_j = k_3
                if j > MAXIMO_ITERACOES:
                   break

        y.append(y[-1] + h_n*((1/6)*k_1 + (2/3)*k_2 + (1/6)*k_3))

    return t, y

def lobatoIIIC(t_0, T, h_n, f, y_0, ordem):
    if(ordem == 2):
        return _lobatoIIIC_ordem2(t_0, T, h_n, f, y_0)
    else:
        return _lobatoIIIC_ordem4(t_0, T, h_n, f, y_0)

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
        malha__do__tempo, aproximacao__numerica = lobatoIIIC(INICIO_INTERVALO, FIM_INTERVALO, h_n, f, CONDICAO_INICIAL, 4)

        # norma euclidiana do erro de discretização global
        e = np.linalg.norm(y_e(FIM_INTERVALO) - aproximacao__numerica[-1])

        # aproximação da ordem do método
        ordem__p = np.log(abs(e__anterior/e))/np.log(FATOR_MULTIPLICATIVO) if caso != 1 else 0

        print("%s %5d %s & %s %9.3e %s & %s %9.3e %s & %s %9.3e %s  \\\\" % ("$", n, "$", "$", h_n, "$", "$",e, "$", "$",ordem__p, "$"));

        e__anterior = e

        # traçado de gráficos para determinados casos
        if caso == 5 or caso == 6 or caso == 8:
            ax1.plot(malha__do__tempo, [y[0] for y in aproximacao__numerica], color="black", linestyle=linestyles[caso%4], label="n = {}".format(str(n)))
            ax2.plot(malha__do__tempo, [y[1] for y in aproximacao__numerica], color="black", linestyle=linestyles[caso%4], label="n = {}".format(str(n)))



    ax1.plot(malha__do__tempo, y_e(malha__do__tempo)[0], color="black", linestyle="solid", label="solução exata $y_1(t)$")
    ax2.plot(malha__do__tempo, y_e(malha__do__tempo)[1], color="black", linestyle="solid", label="solução exata $y_2(t)$")

    ax1.set_xlabel("t (adimensional)")
    ax1.set_ylabel("variável de estado (adimensional)")
    ax1.set_title("Método do Trapézio Implícito para a variável de estado $y_1$ \n(solução exata $y_1(t) = e^{-t}*cos(t)$)", size=12)
    ax1.legend(loc="upper right",)

    ax2.set_xlabel("t (adimensional)")
    ax2.set_ylabel("variável de estado (adimensional)")
    ax2.set_title("Método do Trapézio Implícito para a variável de estado $y_2$ \n(solução exata $y_2(t) = 6e^{-t}*sin(t)$)", size=12)
    ax2.legend(loc="upper right")

    ax1.set_box_aspect(1/2)
    ax2.set_box_aspect(1/2)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
