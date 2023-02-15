# autor: Igor Pontes Tresolavy
# autor: Thiago Antice Rodrigues de Souza

"""
Confeccionando tabelas de convergência para os problemas manufaturados:
y_e_1(t) = e^(-5t)cos(t)
y_e_2(t) = e^(-t) - e^(-4t)

Problema de Cauchy 1D
- d[y(t)]/dt = f(t, y) = -5y - e^(-5t)sin(t)
- y(0) = 1

Problema de Cauchy 2D
d[(y1(t), y2(t))]/dt = ((0, 1), (-4, -5))*(y1(t), y2(t))
(y1(0), y2(0)) = (0, 3)
"""

import matplotlib.pyplot as plt
import numpy as np

# definições
NO_DE_CASOS = 20
INICIO_INTERVALO = 0        #\
CONDICAO_INICIAL_1 = 1      #| condições iniciais
CONDICAO_INICIAL_2 = [0, 3] #/
FIM_INTERVALO = (3.0*np.pi)/4.0
QNTD_PASSOS_INICIAL = 4
FATOR_MULTIPLICATIVO = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL

CASO = 0 # usado para distinguir condições inciais
CONDICOES_INICIAIS = [CONDICAO_INICIAL_1, CONDICAO_INICIAL_2]


# definindo a sequência de passos no tempo e
# a variável de estado
t = np.arange(INICIO_INTERVALO, FIM_INTERVALO + PASSO_INICIAL, PASSO_INICIAL)
y_k = [np.array(CONDICOES_INICIAIS[CASO])]

# Método de Euler Explícito
def euler(T, h_n, f):
    global t, y_k

    t_0 = INICIO_INTERVALO

    t = np.arange(t_0, T + h_n, h_n)
    y_k = [np.array(CONDICOES_INICIAIS[CASO])]

    for t_k in t[1:]:
        y_k.append(y_k[-1] + h_n*f(t_k, y_k[-1]))

    return y_k[-1]

# Solução exata do Problema de Cauchy 1D
def y_e_1(T):
    return np.cos(T)*np.exp(-5*T)

# Solução exata do Problema de Cauchy 2D
def y_e_2(T):
    return np.array([np.exp(-T) - np.exp(-4*T), -np.exp(-T) + 4*np.exp(-4*T)])

# f(t,y) do problema de Cauchy 1D
def f_1(t, y):
    return -5*y - np.exp(-5*t)*np.sin(t)

# f(t,y_1, y_2) do problema de Cauchy 2D
def f_2(t, y):
    return np.array([y[1], -4*y[0] -5*y[1]])

# Erro de Discretização Global
def global_error(T, h_n, y_e, numerical_approximation, f):
    return y_e(T) - numerical_approximation(T, h_n, f)

def main():
    global CASO, t, y_k

    y_e = [y_e_1, y_e_2]
    f = [f_1, f_2]
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7,10))
    fig.tight_layout()
    linestyles = ["dotted", "dashed", "dashdot", (0, (10, 3))]

    print("TABELA DE VERIFICAÇÃO DE SOLUÇÃO MANUFATURADA (1D e 2D)");

    for i in range(0, 2):
        print("\n%dD" %(i+1));

        for caso in range(1, NO_DE_CASOS + 1):
            n = QNTD_PASSOS_INICIAL*(FATOR_MULTIPLICATIVO**(caso-1))
            h_n = (FIM_INTERVALO - INICIO_INTERVALO)/n
            e = np.max(abs(global_error(FIM_INTERVALO, h_n, y_e[i], euler, f[i]))) # norma do máximo
            ordem__p = np.log(abs(e__anterior/e))/np.log(FATOR_MULTIPLICATIVO) if caso != 1 else 0

            print("%s %5d %s & %s %9.3e %s & %s %9.3e %s & %s %9.3e %s  \\\\" % ("$", n, "$", "$", h_n, "$", "$",e, "$", "$",ordem__p, "$"));

            h_n__anterior = h_n
            e__anterior = e

            if caso == 2 or caso == 3 or caso == 8:
                if len(y_k[0].shape) == 0:
                    ax1.plot(t, y_k, color="black", linestyle=linestyles[caso%4], label="$\eta(t,h)$ para n = {}".format(str(n)) )
                else:
                    ax2.plot(t, [y[0] for y in y_k], color="black", linestyle=linestyles[caso%4], label="$\eta_1(t,h)$ para n = {}".format(str(n)))
                    ax3.plot(t, [y[1] for y in y_k], color="black", linestyle=linestyles[caso%4], label="$\eta_2(t,h)$ para n = {}".format(str(n)))


        CASO = CASO + 1

    ax1.plot(t, y_e_1(t), color="black", linestyle="solid", label="solução exata")
    ax2.plot(t, y_e_2(t)[0], color="black", linestyle="solid", label="solução exata")
    ax3.plot(t, y_e_2(t)[1], color="black", linestyle="solid", label="primeira derivada da solução exata")

    ax1.set_xlabel("t (adimensional)")
    ax1.set_ylabel("variável de estado (adimensional)")
    ax1.set_title("Método de Euler 1D para solução exata $y_e(t) = e^{-5t}cos(t)$", size=12)
    ax1.legend(loc="upper right",)

    ax2.set_xlabel("t (adimensional)")
    ax2.set_ylabel("variável de estado (adimensional)")
    ax2.set_title("Método de Euler 2D para a variável de estado $y_1$ \n(solução exata $y_e(t) = e^{-t} - e^{-4t}$)", size=12)
    ax2.legend(loc="upper right",)

    ax3.set_xlabel("t (adimensional)")
    ax3.set_ylabel("variável de estado (adimensional)")
    ax3.set_title("Método de Euler 2D para a variável de estado $y_2$ \n(derivada da solução exata $\dot{y}_e(t) = -e^{-t} + 4e^{-4t}$)", size=12)
    ax3.legend(loc="upper right")

    ax1.set_box_aspect(1/2)
    ax2.set_box_aspect(1/2)
    ax3.set_box_aspect(1/2)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
