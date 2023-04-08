# ative se desejar informações mais detalhadas durante execução
# do algoritmo
DEBUG = 0

# inputs do main.py e utils.py
INICIO_INTERVALO = 0
CONDICAO_INICIAL = [1, 0]
FIM_INTERVALO = 10
QNTD_PASSOS_MMQ = 2**10
PASSO_DE_INTEG_MMQ = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_MMQ
PERTURBACAO = 0.6 # perturbação dos parâmetros do modelo

# inputs do modelo de marketplace
MAX_B = 100000 # quantidade máxima de compradores
MAX_V =  50000 # quantidade máxima de fornecedores
NO_DE_CASOS_MARKETPLACE = 6
QNTD_PASSOS_INICIAL_MARKETPLACE = 128 # da verificação do método de Lobato IIIC
FATOR_MULTIPLICATIVO_MARKETPLACE = 2 # entre cada caso da tabela de convergência

# inputs do mmq.py
NO_DE_PASSOS = 5
# perturbação utilizada na aproximação da matriz jacobiana pelo
# método de diferenças finitas
DP = 2e-4
TOL_MMQ = 2e-8 # tolerância do critério de parada do Gauss-Newton

# inputs do lobato.py
NO_DE_CASOS = 8
TOL = 2e-10
QNTD_PASSOS_INICIAL = 2 # da verificação do método de Lobato IIIC
FATOR_MULTIPLICATIVO = 2 # entre cada caso da tabela de convergência
PASSO_INICIAL = (FIM_INTERVALO - INICIO_INTERVALO)/QNTD_PASSOS_INICIAL
MAXIMO_ITERACOES = 15
