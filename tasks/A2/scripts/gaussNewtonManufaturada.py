# autor: Igor Pontes Tresolavy
# autor: Thiago Antici Rodrigues de Souza

import matplotlib.pyplot as plt
import numpy as np

DEBUG = 0

# parâmetros
NO_DE_PASSOS = 5
PARAMETROS_INICIAIS = np.array([-0.9,-1.0/5,5.0,-0.8])
INICIO_INTERVALO = 0
CONDICAO_INICIAL = [1, 0]
FIM_INTERVALO = 10
QNTD_PASSOS_MMQ = 2**10
PASSO_DE_INTEG_MMQ = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_MMQ
PERTURBACAO = 2e-4
TOL_MMQ = 2e-8
TOL = 2e-8
MAXIMO_ITERACOES = 3

# Solução exata do Problema de Cauchy 2D
def y_e(T):
    return np.array([np.exp(-T)*np.cos(T), 6*np.exp(-T)*np.sin(T)])

# f(t,y_1, y_2) do problema de Cauchy 2D
def f_parametros(param):
    return lambda t, y : np.array([param[0]*y[0] + param[1]*y[1], param[2]*y[0] +param[3]*y[1]])

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



def funcao_de_custo(A, B):
    m, n = A.shape[0], A.shape[1]
    n2, p = B.shape[0], B.shape[1]
    if n2 != n:
       raise ValueError("dimensões incompatíveis")

    if(len(B.shape) == len(A.shape)):
        C = np.zeros((m, p))
    elif(len(B.shape) > len(A.shape)):
        C = np.zeros((m, p, B.shape[2]))
    else:
        C = np.zeros(m)
        for i in range(m):
            for k in range(n):
                C[i] += np.dot(A[i, k],B[k])
        if DEBUG:
            print("C: " + str(C))
        return C

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += np.dot(A[i, k],B[k, j])

    if DEBUG:
        print("C: " + str(C))
    return C



def main():

    # inicialização do algoritmo de aproximação de parâmetros de EDOs pelo método de Gauss-Newton
    malha__do__tempo = np.arange(INICIO_INTERVALO, FIM_INTERVALO + PASSO_DE_INTEG_MMQ, PASSO_DE_INTEG_MMQ)
    data = []
    for t in malha__do__tempo:
       data.append(y_e(t)) 
    
    if DEBUG:
        print("data: " + str(data))

    parametros = PARAMETROS_INICIAIS
    if DEBUG:
        print("parametros: " + str(parametros))

    # Algoritmo de Gauss-Newton
    for i in range(1, NO_DE_PASSOS + 1):

        malha__do__tempo, aproximacao__numerica = lobatoIIIC(INICIO_INTERVALO, FIM_INTERVALO, PASSO_DE_INTEG_MMQ, f_parametros(parametros), CONDICAO_INICIAL, 4)
        if DEBUG:
            print("aproximacao: " + str(aproximacao__numerica))

        # TODO: check if this vector doesn't need to be transposed
        residuo = (np.array(data[1:]) - np.array(aproximacao__numerica[1:]))
        if DEBUG:
            print("residuo: " + str(residuo))

        matriz_jacobiana = np.zeros((QNTD_PASSOS_MMQ, len(parametros), np.array([CONDICAO_INICIAL]).ndim))
        for j in range(0, len(parametros)):
            perturbacao = np.zeros(len(parametros))
            perturbacao[j] = PERTURBACAO
            if DEBUG:
                print("perturbacao: " + str(perturbacao))
            _, tmp_aprox_num = lobatoIIIC(INICIO_INTERVALO, FIM_INTERVALO, PASSO_DE_INTEG_MMQ, f_parametros(parametros + perturbacao), CONDICAO_INICIAL, 4)
            if DEBUG:
                print("tmp_aproximacao: " + str(tmp_aprox_num))
            # print(str(np.array(tmp_aprox_num).shape) + ", " + str(np.array(aproximacao__numerica).shape))
            matriz_jacobiana[:,j] = (np.array(tmp_aprox_num[1:]) - np.array(aproximacao__numerica[1:]))/PERTURBACAO
            if DEBUG:
                print("matriz_jacob: " + str(matriz_jacobiana))
                print("shape matriz_jacob: " + str(matriz_jacobiana.shape))

        p_k = funcao_de_custo(funcao_de_custo(np.linalg.inv(funcao_de_custo(np.transpose(matriz_jacobiana,axes=(1,0,2)),matriz_jacobiana)),np.transpose(matriz_jacobiana,axes=(1,0,2))),residuo)
        parametros += p_k
        print(parametros)

if __name__ == "__main__":
    main()
