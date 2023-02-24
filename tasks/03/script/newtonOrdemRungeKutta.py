import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

# valor de incremento para iteração
# ao longo do círculo unitário do
# fator de amplificação que
# define a região de estabilidade
# do método
INCREMENTO = (2*np.pi)/100

# número máximo de iterações do
# método de newton
MAX = 20


def regiaoDeEstabilidade(fatorDeAmplificacao):

    # teta para a iteração ao longo do círculo unitário
    # que delimita os valores de fronteira da região de
    # estabilidade
    teta = 0  # rad

    # vetor que armazena vetores da região
    # de estabilidade
    regiao = []

    while teta < 2*np.pi:

        psi = fatorDeAmplificacao.copy()

        # enquanto todas as raízes não forem encontradas
        while psi.degree() != 0:

            lambda_h = -0.5 - 0.5j  # chute inicial

            # Método de Newton
            i = 0
            while i < MAX:
                raiz_lambda_h = lambda_h - \
                    (psi(lambda_h) - np.exp(1j*teta))/(psi.deriv()(lambda_h))
                lambda_h = raiz_lambda_h
                i = i + 1

            regiao.append(raiz_lambda_h)

            # diminui grau do polinômio
            psi = psi//(P([-1*raiz_lambda_h, 1])) + \
                psi % (P([-1*raiz_lambda_h, 1]))

        teta = teta + INCREMENTO

    return regiao


def main():

    # polinômios dos fatores de amplificação dos métodos
    # de Runge-Kutta até ordem 4
    RK11 = P([1, 1])
    RK22 = P([1, 1, 1/2])
    RK33 = P([1, 1, 1/2, 1/6])
    RK44 = P([1, 1, 1/2, 1/6, 1/24])

    # Preparando gráficos para traçados
    fig, ax = plt.subplots(1, figsize=(5, 10))
    fig.tight_layout()

    # obtendo regiões de estabilidade para
    # cada método de Runge_Kutta
    regiaoRK11 = regiaoDeEstabilidade(RK11)
    regiaoRK22 = regiaoDeEstabilidade(RK22)
    regiaoRK33 = regiaoDeEstabilidade(RK33)
    regiaoRK44 = regiaoDeEstabilidade(RK44)

    ordenarPontos(regiaoRK11)
    ordenarPontos(regiaoRK22)
    ordenarPontos(regiaoRK33)
    ordenarPontos(regiaoRK44)

    np.sort(regiaoRK22)
    np.sort(regiaoRK33)
    np.sort(regiaoRK44)

    ax.plot([z.real for z in regiaoRK11], [
            z.imag for z in regiaoRK11], color="blue", label="RK11")
    ax.plot([z.real for z in regiaoRK22], [z.imag for z in regiaoRK22],
            color="green", linestyle="solid", label="RK22")
    ax.plot([z.real for z in regiaoRK33], [z.imag for z in regiaoRK33],
            color="orange", linestyle="solid", label="RK33")
    ax.plot([z.real for z in regiaoRK44], [z.imag for z in regiaoRK44],
            color="red", linestyle="solid", label="RK44")
    ax.set_xlabel("$Re(\lambda h)$")
    ax.set_ylabel("$Im(\lambda h)$")
    ax.set_title("Regiões de Estabilidade para Métodos Runge-Kutta")
    ax.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


def transladar_1_positivo(regiao):
    for i in range(len(regiao)):
        novo = np.add(regiao[i], 1)
        regiao[i] = novo


def transladar_1_negativo(regiao):
    for i in range(len(regiao)):
        novo = np.add(regiao[i], -1)
        regiao[i] = novo


def ordenarPontos(regiao):
    transladar_1_positivo(regiao)
    regiao.sort(key=np.angle)
    first_point = regiao[0]
    regiao.append(first_point)
    transladar_1_negativo(regiao)


if __name__ == "__main__":
    main()
