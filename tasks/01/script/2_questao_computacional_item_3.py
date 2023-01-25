# autor: Igor Pontes Tresolavy
# autor: Thiago Antice Rodrigues de Souza

"""
Sabe-se que o Método de Euler possui
erro de ordem de convergência 1.
Portanto, ao se estimar a ordem de convergência
pelos métodos descritos na seção 2.3.2 e 2.3.3
das notas de aula do Professor Roma, espera-se
obter um resultado semelhante ao da depuração
por solução manufaturada (seção 2.3.1).

Ademais, espera-se que a estimativa do erro global também
convirja para 0.

Usou-se o seguinte problema de Lotka-Volterra nesse programa:

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
FIM_INTERVALO = (3.0*np.pi)/4.0
QNTD_PASSOS_INICIAL = 4
FATOR_MULTIPLICATIVO_DA_QUANTIDADE_DE_PASSOS = 2
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL

def f(t, y):
    p = y[0]
    q = y[1]
    a = 0.2
    m = 0.1
    K = 500
    k = 0.2
    b = 0.1
    return np.array([a*p*(1-p/K) - b*p*q/(1+b*p), m*q*(1-1/(k*p))])

def euler(T, h_n, f):
    t_0 = INICIO_INTERVALO

    # definindo a sequência de passos no tempo
    t = np.arange(t_0, T + h_n, h_n)
    y_k = [np.array(CONDICAO_INICIAL)]

    for t_k in t:
        y_k.append(y_k[-1] + h_n*f(t_k, y_k[-1]))

    return y_k[-1]

def global_error_estimate(T, h_n, numerical_approximation, f):
    # ordem de convergência do método de euler = 1
    return numerical_approximation(T, h_n, f) - numerical_approximation(T, h_n/2, f)/(2**1 - 1)

def main():

    print("TABELA DE VERIFICAÇÃO DE UM PROBLEMA DE LOTKA-VOLTERRA");

    for caso in range(1, NO_DE_CASOS + 1):
        n = QNTD_PASSOS_INICIAL*(FATOR_MULTIPLICATIVO_DA_QUANTIDADE_DE_PASSOS**(caso-1))
        h_n = (FIM_INTERVALO - INICIO_INTERVALO)/n
        e = np.linalg.norm(global_error_estimate(FIM_INTERVALO, h_n, euler, f)) # norma euclidiana
        ordem__p = np.log(abs(e__anterior/e))/np.log(FATOR_MULTIPLICATIVO_DA_QUANTIDADE_DE_PASSOS) if caso != 1 else 0

        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n,h_n,e,ordem__p));

        e__anterior = e

if __name__ == "__main__":
    main()
