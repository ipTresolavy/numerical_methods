import numpy as np
from inputs import *

# multiplicação de matriz por matriz ou vetor por vetor
def multMatrixVetor(A, B):
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


def gaussNewton(dados, m, N=NO_DE_PASSOS, t_0=INICIO_INTERVALO, h=PASSO_DE_INTEG_MMQ , t_f=FIM_INTERVALO, y_0=CONDICAO_INICIAL, perturbacao=DP):

    modelo = m.copy()

    print("Passo & ", end='')
    for i in range(1, len(modelo.parametros())):
        print("param%d & " % (i), end='')
    print("param%d\\\\" % (len(modelo.parametros())))

    # Algoritmo de Gauss-Newton
    for i in range(1, N+1):

        aproximacaoNumerica = modelo(t_0=t_0, h=h, t_f=t_f, y_0=y_0)
        if DEBUG:
            print("aproximacao: " + str(aproximacaoNumerica))

        residuo = (np.array(dados[1:]) - np.array(aproximacaoNumerica[1:]))

        if DEBUG:
            print("residuo: " + str(residuo))

        # define dimensões da matriz jacobiana
        matrizJacobiana = np.zeros((len(aproximacaoNumerica)-1, len(modelo.parametros()), np.array([aproximacaoNumerica[0]]).ndim))

        for j in range(0, len(modelo.parametros())):
            dp = np.zeros(len(modelo.parametros()))
            dp[j] = perturbacao

            if DEBUG:
                print("perturbacao: " + str(dp))

            tmpAproxNum = modelo(t_0=t_0, h=h, f=modelo.f(h, param=(modelo.parametros() + dp)), t_f=t_f, y_0=y_0)

            if DEBUG:
                print("Aproximacao perturbada: " + str(tmpAproxNum))

            matrizJacobiana[:,j] = (np.array(tmpAproxNum[1:]) - np.array(aproximacaoNumerica[1:]))/perturbacao

            if DEBUG:
                print("matriz_jacob: " + str(matrizJacobiana))
                print("shape matriz_jacob: " + str(matrizJacobiana.shape))

        p_k = multMatrixVetor(\
                multMatrixVetor(\
                    np.linalg.inv(\
                        multMatrixVetor(\
                            np.transpose(\
                                matrizJacobiana,axes=(1,0,2)\
                            ),\
                            matrizJacobiana\
                        )\
                    ),\
                    np.transpose(\
                        matrizJacobiana,\
                        axes=(1,0,2)\
                    )\
                ),\
                residuo\
            )

        # atualiza parâmetros do modelo
        modelo.setParametros(modelo.parametros() + p_k)

        if abs(np.max(p_k)) < TOL_MMQ:
            break

        print("%5d & " % (i), end='')
        for j in range(0, len(modelo.parametros())-1):
            print("%s %9.3e %s & " % ("$", modelo.parametros()[j], "$"), end='')
        print("%s %9.3e %s\\\\" % ("$", modelo.parametros()[len(modelo.parametros())-1], "$"))

    return modelo.parametros()
