#EDO
#   x'=y, x(pi^0.5)=0
#   y'=y/t-4(t^2)x, y(pi^0.5)=-2pi^0.5
# Sol analitica
#   x=sin(t^2), y=2tcos(t^2)
#-----------------------------------------------------
import math
import matplotlib.pyplot as plt
import numpy as np
#-----------------------------------------------------
# FUNCAO PARA DETERMINAR AS SOLUCOES EXATAS
def SolAnalitica(vt):                   # Def das solucoes analiticas no intervalo vt
    ysol=[ [math.sin(i**2),2*i*math.cos(i**2)] for i in vt]
    return np.array(ysol)
#-------------------------------------------------------    
def f(t,y):                             # Definicao de f
    # y=[y0,y1]: vetor de 1x2 dimensao, t: instante t
    fy1=y[1]
    fy2=y[1]/t-4*y[0]*t**2
    return np.array([fy1,fy2])
#-----------------------------------------------------
#   METODO DE EULER
def Euler(vt,vy0,dt):                  # Methodo de Euler
    # vt: vetor tempo, vy0: vetor inicial, dt: tamanho passo
    ys=vy0.copy()                      # inicailizacao vetor solucao
    for i in range(len(vt)-1):
        ys.append(ys[i]+dt*f(vt[i],ys[i]))
    return np.array(ys)
#----------------------------------------------------
#   METODO EULER IMPLICITO
def EulerImp(vt,vy0,dt):               # npf: iteracoes ponto fixo
    # vt: vetor tempo, vy0: vetor inicial, dt: tamanho passo
    npf=6
    ys=vy0.copy()
    for i in range(len(vt)-1):
        y0_pf=ys[-1]+dt*f(vt[i],ys[-1])   # chute inicial(usando Euler) para usar ponto fixo
        yPonFix=y0_pf.copy()              # Inicio Methodo ponto fixo
        for j in range(npf):
            yPonFix=ys[-1]+dt*f(vt[i]+dt,yPonFix) 
        ys.append(yPonFix)                # Fim Metodo ponto Fixo
    return np.array(ys)
#------------------------------------------------
#   METODO TRAPEZIO IMPLICITO
def TraImp(vt,vy0,dt):
    npf=6
    ys=vy0.copy()
    for i in range(len(vt)-1):
        y0_pf=ys[-1]+dt*f(vt[i],ys[-1]) # chute inicial(usando Euler) para usar ponto fixo
        yPonFix=y0_pf.copy()            # Inicio Methodo ponto fixo
        for j in range(npf):
            yPonFix=ys[-1]+dt*(f(vt[i],ys[-1])+f(vt[i]+dt,yPonFix))/2
        ys.append(yPonFix)              # Fim Metodo ponto Fixo
    return np.array(ys)
#------------------------------------------------------
#   MAIN INPUT
n_metodo= int(input("Escreva um número 1(Euler), 2(Euler Implícito), 3(Trapézio Implícito) :"))
T=float(input("O intervalo da solução será [sqrt(pi) , T], escreva T: "))
print(f"A seguir, o programa apresentara os resultados usando o Método {n_metodo} \
no intervalo [sqrt(pi) , {T}] para os passos de integração h=(T-sqrt(pi)/2^m,\
com m=5,6,..14,15) e gráficos para m=5,9,11")
#------------------------------------------------------
#   SOLUCAO EXATA PARA O INTERVALO [t0, T]
t0=math.sqrt(math.pi)               # para o intervalo [t0 tf]
m=[5,6,7,8,9,10,11,12,13,14,15]
ta=np.linspace(t0,T,2**m[-1])       # intervalo de tempo p solucao analitica
y_analitica=SolAnalitica(ta)        # vetor solucao analitica(exata)
#-------------------------------------------------------
#   SOLUCOES NUMERICAS PARA m=5,6,...,14,15
if n_metodo==1:
    metodotitle='Euler'
    funcmetodo=Euler
elif n_metodo==2:
    metodotitle='Euler Implícito'
    funcmetodo=EulerImp
else:
    metodotitle='Trapézio Implícito'
    funcmetodo=TraImp

vsol=[]
ts=[]       # vetor tempos para cada caso
for i in range(len(m)):
    n=2**m[i]
    h=(T-t0)/n
    t=np.linspace(t0,T,n+1)              # vetor instantes de tempo 
    y_0=[np.array([0,-2*math.sqrt(math.pi)])]
    ysol=funcmetodo(t,y_0,h)
    vsol.append(ysol)
    ts.append(t)
#--------------------------------------------------------------
#   GRAFICOS PARA COMPARAR ALGUMAS SOLUCOES COM A SOL EXATA
namepic= 'metodo_%d' % n_metodo
plot1 = plt.figure(1)      # comparando graficos da variavel x
plt.plot(ts[0],vsol[0][:,0],'k+',linewidth=1.2,label = 'n = %d' %2**m[0])
plt.plot(ts[4],vsol[4][:,0],'ko',markersize=0.8,label = 'n = %d' %2**m[4])
plt.plot(ts[6],vsol[6][:,0],'k--',markersize=0.5,linewidth=0.6,label = 'n = %d' %2**m[6])
plt.plot(ta,y_analitica[:,0], c = 'k',linewidth=0.7,label = 'Solução Analítica')
plt.xlabel('t(unidade) - variável independente')
plt.ylabel('x(unidade) - variável de estado')
plt.title('Aproximação Numérica da variavel x usando %s' %metodotitle)
plt.legend(loc='upper left')
#plt.savefig('%sx_T%s.png' % (namepic,T),dpi=300)

plot1 = plt.figure(2)      # comparando graficos da variavel y
plt.plot(ts[0],vsol[0][:,1],'k+',linewidth=1.2,label = 'n = %d' %2**m[0])
plt.plot(ts[4],vsol[4][:,1],'ko',markersize=0.8,label = 'n = %d' %2**m[4])
plt.plot(ts[6],vsol[6][:,1],'k--',markersize=0.5,linewidth=0.6,label = 'n = %d' %2**m[6])
plt.plot(ta,y_analitica[:,1], c = 'k',linewidth=0.7,label = 'Solução Analítica')
plt.xlabel('t(unidade) - variável independente')
plt.ylabel('y(unidade) - variável de estado')
plt.title('Aproximação Numérica da variavel y usando %s' %metodotitle)
plt.legend(loc='upper left')
#plt.savefig('%sy_T%s.png' % (namepic,T),dpi=300)
#---------------------------------------------------------
#   ERROS GLOBAIS NO PONTO T
errosx=[]; errosy=[];q=['-']
for i in range(len(m)):
    errosx.append(abs(vsol[i][-1,0]-y_analitica[-1,0]))  # para x
    errosy.append(abs(vsol[i][-1,1]-y_analitica[-1,1]))  # para y
    
errosnorm=max(errosx,errosy)
q=[errosnorm[i]/errosnorm[i+1] for i in range(len(errosnorm)-1)]
p=[math.log(q[i],2) for i in range(len(q))]
#----------------------------------------------------------------------
#   TABELA COM OS ERROS GLOBAIS, E ESTIMATIVA DA ORDEM DE CONVERGENCIA DO METODO
q.insert(0,0); p.insert(0,0);
print('-----------------------------------------------------------------------------------------')
print('\t Errosx \t\t  Errosy  \t\t  max{|Errosx|,|Errosy|} \t q=Erro_2h/Erro_h \t Log_2(q)')
print('-----------------------------------------------------------------------------------------')
for i in range(len(m)):
    print("%15.6f \t %15.6f \t %15.6f \t %15.6f \t %15.6f" %(errosx[i],errosy[i],errosnorm[i],q[i],p[i]))
#---------------------------------------------------------------------
#   ESTIMATIVA NUMERICA DA ORDEM DE CONVERGENCIA SEM SOL ANALITICA
normq1=[];normq2=[];
for i in range(len(m)-2):
    normq1.append(max(abs(vsol[i+1][-1,:]-vsol[i][-1,:])))
    normq2.append(max(abs(vsol[i+2][-1,:]-vsol[i+1][-1,:])))
    
qn=[normq1[i]/normq2[i] for i in range(len(normq1))]
pn=[math.log(qn[i],2) for i in range(len(qn))]  # ordem de convergencia estimado numericamente
pn.insert(0,0);pn.insert(0,0);
print('==================================================================================')
print('\t Estimação Numérica da ordem de convergencia do Método %s' %metodotitle)
print('----------------------------------------------------------------------------------')
print('  m\t\t n=2^m\t h=(T-sqrt(pi))/n  \t y_n(t,h)=(y1,y2)\t log_2(||y_n(t,2h)-y_n(t,h)||/||y_n(t,h)-y_n(t,h/2)||)')
print('==================================================================================')
for i in range(len(m)):
    print("%4d\t %7d\t %10.5f\t (%10.5f, %10.5f)\t %10.5e" %(m[i],2**m[i],(T-t0)/(2**m[i]),vsol[i][-1,:][0],vsol[i][-1,:][1],pn[i]))
    