import numpy as np
from lobatto import lobattoIIIC, verificarModeloComSolucaoConhecida, verificarMarketplace
from mmq import gaussNewton
from utils import *
from inputs import *

def verificacaoPorSolucaoManufaturada():

    # instancia solução manufaturada
    y_e = Y_e()

    print("Parâmetros reais: " + str(y_e.parametros())) 

    # inicialização do algoritmo de aproximação de parâmetros de EDOs pelo método de Gauss-Newton
    dados = obterDados(y_e)

    print("\nt & ", end='')
    print("$ y_1(t,\\boldsymbol{x}) $ & ", end='')
    print("$ y_2(t,\\boldsymbol{x}) $ \\\\\a")
    for t in range(len(dados[0::100])):
        print("%s %4d   %s & " % ("$", 100*t, "$"), end='')
        print("%s %9.3e %s & " % ("$", dados[100*t][0], "$"), end='')
        print("%s %9.3e %s \\\\" % ("$", dados[100*t][1], "$"))



    # perturba parâmetros para usá-los como estimativa inicial do método
    # param = perturbarParametros(y_e)
    param = [-1.16266218, -0.15810197, 8.70269332, -0.40795276]
    modelo = Y(param)

    print("\nEstimativa inicial dos parâmetros: " + str(param))

    # aplica Gauss-Newton
    param = gaussNewton(dados, modelo)
    modelo.setParametros(param)

    print("\nResultado de GaussNewton:\t" + str(param) + "\n")

    print("Tabela de Convergência para o Método de Lobatto IIIC:")

    verificarModeloComSolucaoConhecida(y_e, modelo)


# Solução manufaturada
class Y_e():
    def __init__(self):
        self.__parametros = [-1,-1/6,6,-1]

    # f(t,y_1, y_2) do problema de Cauchy 2D
    def f(self):
        return lambda t, y : np.array([self.__parametros[0]*y[0] + self.__parametros[1]*y[1], self.__parametros[2]*y[0] +self.__parametros[3]*y[1]])

    def parametros(self):
        return self.__parametros

    def setParametros(self, parametros):
        self.__parametros = parametros

    def __call__(self, T=np.arange(INICIO_INTERVALO, FIM_INTERVALO + PASSO_DE_INTEG_MMQ, PASSO_DE_INTEG_MMQ)):
        return np.array([np.exp(-T)*np.cos(T), 6*np.exp(-T)*np.sin(T)])


# Modelo utilizado para cálculo dos parâmetros e discretização do modelo
class Y():
    def __init__(self,parametros):
        self.__parametros = parametros

    # f(t,y_1, y_2) do problema de Cauchy 2D
    def f(self, h, param=None):
        if param is None:
            param = self.__parametros
        return lambda t, y : np.array([param[0]*y[0] + param[1]*y[1], param[2]*y[0] +param[3]*y[1]])

    def __call__(self, metodoNumerico=lobattoIIIC, t_0=INICIO_INTERVALO, h=PASSO_DE_INTEG_MMQ, f=None, t_f=FIM_INTERVALO, y_0=CONDICAO_INICIAL):
        if f is None:
            f = self.f(h)
        return metodoNumerico(t_0, t_f, h, f, y_0, 4)[1]

    def parametros(self):
        return self.__parametros

    def setParametros(self, parametros):
        self.__parametros = parametros

    def copy(self):
        return Y(self.__parametros)


def modeloDeMarketplace():
    # instancia modelo de Marketplace
    parametrosIniciais = [
                            0.03,   # p_B 
                            0.4,    # q_B
                            0.01,   # gamma_B
                            0.5,    # alpha_B
                            0.1,    # beta_B
                            0.02,   # p_V
                            0.3,    # q_V
                            0.02,   # gamma_V
                            0.3,    # alpha_V
                            0.2,    # beta_V
                            0.00001 # k
                        ]
    modelo_e = Marketplace(parametrosIniciais, precoExponencial, publicidadeLinear)

    print("Parâmetros reais: " + str(modelo_e.parametros())) 

    # inicialização do algoritmo de aproximação de parâmetros de EDOs pelo método de Gauss-Newton
    t_f = 20
    n = 2**10
    dados = modelo_e(t_0=0, h=t_f/n, t_f=t_f, y_0=[0,0])
   
    # perturba parâmetros para usá-los como estimativa inicial do método
    param = perturbarParametros(modelo_e,porcentagem=0.20)
    # param = [2.95463844e-02,4.36487449e-01,8.92674823e-03,5.73971093e-01,\
    #          1.14448375e-01,2.38792845e-02,2.86940598e-01,1.62183356e-02,\
    #          2.76638789e-01,2.23807143e-01,9.93812423e-06]
    # param = [2.72755804e-02,3.90750161e-01,7.27950147e-03,4.62749593e-01,\
    #          1.00771391e-01,1.93516514e-02,2.23064845e-01,2.41850508e-02,\
    #          3.48081945e-01,1.77421822e-01,1.03287074e-05]
    # param = [3.28060823e-02,4.10469077e-01,1.19613572e-02,5.13892706e-01,\
    #  1.09366351e-01,1.69138571e-02,2.63875243e-01,1.70391657e-02,\
    #  3.18524777e-01,1.90113773e-01,8.13892255e-06]
    param = [2.48972907e-02,3.84929223e-01,1.14376325e-02,4.99074190e-01,\
             1.05749633e-01,2.09003018e-02,2.83413625e-01,1.64831541e-02,\
             3.12564945e-01,2.27612676e-01,9.21041546e-06]
    modelo = Marketplace(param, precoExponencial, publicidadeLinear)
    
    print("Estimativa inicial dos parâmetros: " + str(param))
   
    # aplica Gauss-Newton
    n = 2**10
    param = gaussNewton(dados, modelo, N=15, t_0=0, h=t_f/n, t_f=t_f, y_0=[0,0], perturbacao=1e-10, step=1)
    modelo.setParametros(param)
    
    print("Resultado de GaussNewton: " + str(param))
   
    print("Tabela de Convergência para o Método de Lobatto IIIC:\n")

    solucao = verificarMarketplace(modelo, t_0=0, t_f=t_f, y_0=[0,0])
    # print(np.argwhere(np.array([y[0] for y in solucao]) >= MAX_B/2))
    # print(np.argwhere(np.array([y[1] for y in solucao]) >= MAX_V/2))
    modelo = Marketplace(parametrosIniciais, precoLogistico, publicidadeLinear)
    solucao = verificarMarketplace(modelo, t_0=0, t_f=t_f, y_0=[0,0])
    # print(np.argwhere(np.array([y[0] for y in solucao]) >= MAX_B/2))
    # print(np.argwhere(np.array([y[1] for y in solucao]) >= MAX_V/2))
    

# Modelo utilizado para cálculo dos parâmetros e discretização do modelo
class Marketplace():
    def __init__(self,parametros, preco, publicidade):
        self.__parametros = parametros
        self.__preco = preco
        self.__publicidade = publicidade

    def preco(self):
        return self.__preco

    def setPreco(self, novo_preco):
        self.__preco = novo_preco

    def publicidade(self):
        return self.__publicidade

    def setPublicidade(self, nova_publicidade):
        self.__publicidade = nova_publicidade
        
    def f(self, h, param=None, preco=None, publicidade=None):
        if param is None:
            param = self.__parametros
        if preco is None:
            preco = self.__preco
        if publicidade is None:
            publicidade = self.__publicidade

        p_b     = param[ 0]
        q_b     = param[ 1]
        gamma_b = param[ 2]
        alpha_b = param[ 3]
        beta_b  = param[ 4]
        p_v     = param[ 5]
        q_v     = param[ 6]
        gamma_v = param[ 7]
        alpha_v = param[ 8]
        beta_v  = param[ 9]
        k       = param[10]
        return lambda t, y : np.array([(p_b + q_b * (y[0]/MAX_B) + gamma_b * (y[1] / MAX_V)) * (MAX_B - y[0]) *\
                                        (1 - alpha_b * (preco(t)-preco(t-h))/preco(t-h) + beta_b * max(0,(publicidade(t) - publicidade(t-h))/publicidade(t-h))),\
                                       (p_v + q_v * (y[1]/MAX_V) + gamma_v * (y[0]/MAX_B)) * (MAX_V - y[1]) *\
                                        (1 + alpha_v * (preco(t)-preco(t-h))/preco(t-h) + beta_v * max(0,(publicidade(t) - publicidade(t-h))/publicidade(t-h))) - k*y[1]*y[1]])

    def __call__(self, metodoNumerico=lobattoIIIC, t_0=INICIO_INTERVALO, h=PASSO_DE_INTEG_MMQ, f=None, t_f=FIM_INTERVALO, y_0=CONDICAO_INICIAL):
        if f is None:
            f = self.f(h=h)
        return metodoNumerico(t_0, t_f, h, f, y_0, 4)[1]

    def parametros(self):
        return self.__parametros

    def setParametros(self, parametros):
        self.__parametros = parametros

    def copy(self):
        return Marketplace(self.__parametros, self.__preco, self.__publicidade)

def precoExponencial(t, precoInicial=5, precoFinal=20, constanteDeTempo=5):
    return precoFinal + (precoInicial - precoFinal)*np.exp(-t/constanteDeTempo)

def precoLogistico(t, precoInicial=5, precoFinal=20, taxaDeCrescimento=5):
    t_0 = (1/taxaDeCrescimento) * np.log(precoFinal/precoInicial - 1)
    return precoFinal/(1 + np.exp(-taxaDeCrescimento * (t - t_0)))

def publicidadeLinear(t, publicidadeInicial=5, taxaDeCrescimento=1):
    return publicidadeInicial + taxaDeCrescimento * t

if __name__ == "__main__":
    while True:
        opcao = input("Digite 1 para resultados da verificação por solução manufaturada\ne 2 para modelo de marketplace (q para sair) = ")
        if opcao == 'q':
            break
        elif opcao == '1':
            verificacaoPorSolucaoManufaturada()
        elif opcao == '2':
            modeloDeMarketplace()
