import numpy as np
from lobato import lobatoIIIC, verificarModeloComSolucaoConhecida, verificarMarketplace
from mmq import gaussNewton
from utils import *
from inputs import *

def verificacaoPorSolucaoManufaturada():

    # instancia solução manufaturada
    y_e = Y_e()

    print("Parâmetros reais: " + str(y_e.parametros())) 

    # inicialização do algoritmo de aproximação de parâmetros de EDOs pelo método de Gauss-Newton
    dados = obterDados(y_e)

    # perturba parâmetros para usá-los como estimativa inicial do método
    # param = perturbarParametros(y_e)
    param = [-1.16266218, -0.15810197, 8.70269332, -0.40795276]
    modelo = Y(param)

    print("\nEstimativa inicial dos parâmetros: " + str(param))

    # aplica Gauss-Newton
    param = gaussNewton(dados, modelo)
    modelo.setParametros(param)

    print("\nResultado de GaussNewton:\n\t" + str(param))

    print("Tabela de Convergência para o Método de Lobato IIIC:\n")

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

    def __call__(self, metodoNumerico=lobatoIIIC, t_0=INICIO_INTERVALO, h=PASSO_DE_INTEG_MMQ, f=None, t_f=FIM_INTERVALO, y_0=CONDICAO_INICIAL):
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
    # parametrosIniciais = [0.049512924, 0.095579906, 0.025336629, 0.028506817, 0.027093797, 0.079573898,\
    #                       0.012027065, 0.058315987, 0.035072117, 0.068207706, 0.0027021291]
    # parametrosIniciais = [4.95129241e-02, 9.55799060e-02, 2.53366289e-02, 1.62442210e+01,\
    #                       1.62428067e+01, 7.95738981e-02, 1.20270647e-02, 5.83159869e-02,\
    #                       4.04535015e+03, 4.04524687e+03, 2.70212910e-03]
    # parametrosIniciais = [4.95129241e-03,9.55799060e-02,2.53366289e-02,2.96214093e+01,\
    #                       2.96199950e+01,7.95738981e-03,1.20270647e-02,5.83159869e-02,\
    #                       4.51468962e+03,3.57590740e+03,2.70212910e-03]
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
    n = 2**15
    dados = modelo_e(t_0=0, h=t_f/n, t_f=t_f, y_0=[0,0])
   
    # perturba parâmetros para usá-los como estimativa inicial do método
    param = perturbarParametros(modelo_e,porcentagem=0.03)
    modelo = Marketplace(param, precoExponencial, publicidadeLinear)
    
    print("Estimativa inicial dos parâmetros: " + str(param))
   
    # aplica Gauss-Newton
    param = gaussNewton(dados, modelo, t_0=0, h=t_f/n, t_f=t_f, y_0=[0,0], perturbacao=1e-12)
    modelo.setParametros(param)
    
    print("Resultado de GaussNewton: " + str(param))
   
    print("Tabela de Convergência para o Método de Lobato IIIC:\n")

    verificarMarketplace(modelo_e, t_0=0, t_f=t_f, y_0=[0,0])
    modelo_e = Marketplace(parametrosIniciais, precoLogistico, publicidadeLinear)
    verificarMarketplace(modelo_e, t_0=0, t_f=t_f, y_0=[0,0])

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

    def __call__(self, metodoNumerico=lobatoIIIC, t_0=INICIO_INTERVALO, h=PASSO_DE_INTEG_MMQ, f=None, t_f=FIM_INTERVALO, y_0=CONDICAO_INICIAL):
        if f is None:
            f = self.f(h=h)
        return metodoNumerico(t_0, t_f, h, f, y_0, 4)[1]

    def parametros(self):
        return self.__parametros

    def setParametros(self, parametros):
        self.__parametros = parametros

    def copy(self):
        return Marketplace(self.__parametros, self.__preco, self.__publicidade)

def precoExponencial(t, precoInicial=5, precoFinal=20, constanteDeTempo=1):
    return precoFinal + (precoInicial - precoFinal)*np.exp(-t/constanteDeTempo)

def precoLogistico(t, precoInicial=5, precoFinal=20, taxaDeCrescimento=1):
    t_0 = (1/taxaDeCrescimento) * np.log(precoFinal/precoInicial - 1)
    return precoFinal/(1 + np.exp(-taxaDeCrescimento * (t - t_0)))

def publicidadeLinear(t, publicidadeInicial=5, taxaDeCrescimento=0.1):
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
