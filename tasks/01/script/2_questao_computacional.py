# autor: Igor Pontes Tresolavy
# autor: Thiago Antice Rodrigues de Souza

"""
Confeccionando tabelas de convergência para o problema manufaturado:
y_e(t) = e^(-5t)cos(t)

Problema de Cauchy
- d[y(t)]/dt = f(t, y) = -5y - e^(-5t)sin(t)
- y(0) = 1

"""

import matplotlib.pyplot as plt
import numpy as np

# definições
NO_DE_CASOS = 20
INICIO_INTERVALO = 0 # condições iniciais
CONDICAO_INICIAL = 1
FIM_INTERVALO = (3.0*np.pi)/4.0
QNTD_PASSOS_INICIAL = 4
FATOR_MULTIPLICATIVO_DA_QUANTIDADE_DE_PASSOS = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL

def y_e(T):
    return np.cos(T)*np.exp(-5*T)

def f(t, y):
    return -5*y - np.exp(-5*t)*np.sin(t)

def phi(t, y, h, f):
        return f(t, y)

def numerical_approximation(T, h_n):
    t_0 = INICIO_INTERVALO

    # definindo a sequência de passos no tempo
    t = np.arange(t_0, T + h_n, h_n)
    y_k = [np.array(CONDICAO_INICIAL)]

    for t_k in t:
        y_k.append(y_k[-1] + h_n*phi(t_k, y_k[-1], h_n, f))

    return y_k[-1]

def global_error(T, h_n):
    return y_e(T) - numerical_approximation(T, h_n)

def main():
    print("TABELA DE VERIFICAÇÃO DE SOLUÇÃO MANUFATURADA");

    for caso in range(1, NO_DE_CASOS):
        n = QNTD_PASSOS_INICIAL*(FATOR_MULTIPLICATIVO_DA_QUANTIDADE_DE_PASSOS**(caso-1))
        h_n = (FIM_INTERVALO - INICIO_INTERVALO)/n
        e = abs(global_error(FIM_INTERVALO, h_n))
        ordem__p = np.log(abs(e__anterior/e))/np.log(h_n__anterior/h_n) if caso != 1 else 0

        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n,h_n,e,ordem__p));

        h_n__anterior = h_n
        e__anterior = e

if __name__ == "__main__":
    main()
