import numpy as np
from inputs import *

def obterDados(y):

    malhaDoTempo = np.arange(INICIO_INTERVALO, FIM_INTERVALO + PASSO_DE_INTEG_MMQ, PASSO_DE_INTEG_MMQ)
    dados = []

    for t in malhaDoTempo:
       dados.append(y(t)) 
    
    if DEBUG:
        print("data: " + str(dados))

    return dados

def perturbarParametros(y, porcentagem=PERTURBACAO):

    parametros = y.parametros()
    perturbacao = np.random.uniform(low=-porcentagem, high=porcentagem, size=np.array(parametros).shape)
    parametrosPerturbados = parametros * (1 + perturbacao)

    return parametrosPerturbados


