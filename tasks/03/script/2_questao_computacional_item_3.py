# autor: Igor Pontes Tresolavy
# autor: Thiago Antice Rodrigues de Souza

"""
Sabe-se que o Método de Euler possui
erro de ordem de convergência 1.
Portanto, ao se estimar a ordem de convergência
pelos métodos descritos na seção 2.3.2 e 2.3.3 das notas de aula do Professor Roma, espera-se obter um resultado semelhante ao da depuração
por solução manufaturada (seção 2.3.1).

Ademais, espera-se que a estimativa do erro global também
convirja para 0.

Usou-se o seguinte problema de Lotka-Volterra nesse programa
(Lotka-Volterra com características de relacionamento mutual;
ver https://en.wikipedia.org/wiki/Mutualism_(biology)#Mathematical_modeling):

dp/dt = a*p(1-p/K) - b*p*q/(1+b*p)
dq/dt = m*q(1-1/(k*p))

com a = 0.2, m = 0.1, K= 500, k = 0.2, b = 0.1, p(0) = 10,
q(0) = 5
"""

import matplotlib.pyplot as plt
import numpy as np

# definições
NO_DE_CASOS = 20
INICIO_INTERVALO = 0 # condições iniciais
CONDICAO_INICIAL = [10, 5]
FIM_INTERVALO = 65
QNTD_PASSOS_INICIAL = 4
FATOR_MULTIPLICATIVO = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL

# definindo a sequência de passos no tempo e
# a variável de estado
t = np.arange(INICIO_INTERVALO, FIM_INTERVALO + PASSO_INICIAL, PASSO_INICIAL)
y_k = [np.array(CONDICAO_INICIAL)]

def euler(T, h_n, f):
    global t, y_k

    t_0 = INICIO_INTERVALO

    # definindo a sequência de passos no tempo
    t = np.arange(t_0, T + h_n, h_n)
    y_k = [np.array(CONDICAO_INICIAL)]

    for t_k in t[1:]:
        y_k.append(y_k[-1] + h_n*f(t_k, y_k[-1]))

    return y_k[-1]

def f(t, y):
    p = y[0]
    q = y[1]
    a = 0.2
    m = 0.1
    K = 500
    k = 0.2
    b = 0.1
    return np.array([a*p*(1-p/K) - b*p*q/(1+b*p), m*q*(1-1/(k*p))])

def global_error_estimate(T, h_n, numerical_approximation, f):
    # ordem de convergência do método de euler = 1
    return numerical_approximation(T, h_n, f) - numerical_approximation(T, h_n/2, f)/(2**1 - 1)

def main():
    global t, y_k

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,20))
    fig.tight_layout()
    linestyles = ["dotted", "dashed", "dashdot", (0, (10, 3))]

    print("TABELA DE VERIFICAÇÃO DE UM PROBLEMA DE LOTKA-VOLTERRA");

    for caso in range(1, NO_DE_CASOS + 1):
        n = QNTD_PASSOS_INICIAL*(FATOR_MULTIPLICATIVO**(caso-1))
        h_n = (FIM_INTERVALO - INICIO_INTERVALO)/n
        # norma euclidiana
        e = np.linalg.norm(global_error_estimate(FIM_INTERVALO, h_n, euler, f))
        ordem__p = np.log(abs(e__anterior/e))/np.log(FATOR_MULTIPLICATIVO) if caso != 1 else 0

        print("%s %5d %s & %s %9.3e %s & %s %9.3e %s & %s %9.3e %s  \\\\" % ("$", n, "$", "$", h_n, "$", "$",e, "$", "$",ordem__p, "$"));

        if caso == 4 or caso == 6 or caso == 9 or caso == 19:
            ax1.plot(t, [y[0] for y in y_k], color="black", linestyle=linestyles[caso%4], label="$p_k$ para n = {}".format(str(n)))
            ax2.plot(t, [y[1] for y in y_k], color="black", linestyle=linestyles[caso%4], label="$q_k$ para n = {}".format(str(n)))

        e__anterior = e

    ax1.set_xlabel("t (adimensional)")
    ax1.set_ylabel("variável de estado (adimensional)")
    ax1.set_title("Aproximações de $p(t)$ para diversos valores de $n$", size=12)
    ax1.legend(loc="upper left", fontsize=17)

    ax2.set_xlabel("t (adimensional)")
    ax2.set_ylabel("variável de estado (adimensional)")
    ax2.set_title("Aproximações de $q(t)$ para diversos valores de $n$", size=12)
    ax2.legend(loc="upper left",fontsize=17)

    ax1.set_box_aspect(1/2)
    ax2.set_box_aspect(1/2)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
