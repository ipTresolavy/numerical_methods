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
INICIO_INTERVALO = 0 # condições iniciais
CONDICAO_INICIAL_1 = 1
CONDICAO_INICIAL_2 = [0, 3]
FIM_INTERVALO = (3.0*np.pi)/4.0
QNTD_PASSOS_INICIAL = 4
FATOR_MULTIPLICATIVO_DA_QUANTIDADE_DE_PASSOS = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL

CASO = 0 # usado para definir condições inciais
CONDICOES_INICIAIS = [CONDICAO_INICIAL_1, CONDICAO_INICIAL_2]

def y_e_1(T):
    return np.cos(T)*np.exp(-5*T)

def y_e_2(T):
    return np.array([np.exp(-T) - np.exp(-4*T), -np.exp(-T) + 4*np.exp(-4*T)])

def f_1(t, y):
    return -5*y - np.exp(-5*t)*np.sin(t)

def f_2(t, y):
    return np.array([y[1], -4*y[0] -5*y[1]])

def euler(T, h_n, f):
    t_0 = INICIO_INTERVALO

    # definindo a sequência de passos no tempo
    t = np.arange(t_0, T + h_n, h_n)
    y_k = [np.array(CONDICOES_INICIAIS[CASO])]

    for t_k in t:
        y_k.append(y_k[-1] + h_n*f(t_k, y_k[-1]))

    return y_k[-1]

def global_error(T, h_n, y_e, numerical_approximation, f):
    return y_e(T) - numerical_approximation(T, h_n, f)

def main():
    global CASO

    y_e = [y_e_1, y_e_2]
    f = [f_1, f_2]

    print("TABELA DE VERIFICAÇÃO DE SOLUÇÃO MANUFATURADA (1D e 2D)");

    for i in range(0, 2):
        print("\n%dD" %(i+1));

        for caso in range(1, NO_DE_CASOS + 1):
            n = QNTD_PASSOS_INICIAL*(FATOR_MULTIPLICATIVO_DA_QUANTIDADE_DE_PASSOS**(caso-1))
            h_n = (FIM_INTERVALO - INICIO_INTERVALO)/n
            e = np.max(abs(global_error(FIM_INTERVALO, h_n, y_e[i], euler, f[i])))
            ordem__p = np.log(abs(e__anterior/e))/np.log(h_n__anterior/h_n) if caso != 1 else 0

            print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n,h_n,e,ordem__p));

            h_n__anterior = h_n
            e__anterior = e
        CASO = CASO + 1

if __name__ == "__main__":
    main()
