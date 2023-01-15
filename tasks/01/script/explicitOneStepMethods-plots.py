# -*- coding: utf-8 -*-
""" 
Spyder Editor Spyder 3.2.3 

# MAP5725 - Roma, 01-02/2020.

# General, Explicit One-Step Methods for 
              d^2/dt^2 x(t) = - x(t)
              x(0)=1; d/dt x(0)=0
           
"""

import matplotlib.pyplot as plt
import numpy as np
#############################################################################
def phi(t,y,dt,f):
    # define discretization function 
    
    k1 = f(t, y)
    k2 = f(t+dt/2, y + dt/2*k1)
    k3 = f(t+dt/2, y + dt/2*k2)
    k4 = f(t+dt, y + dt*k3)
    
    #return 1/6*(k1 + 2*k2 + 2*k3 + k4)     # classical RK-44
    return k1    # euler method
############################################################################
############################################################################
def f(t, y):
    # input n-dim ode system right hand side: f=(f0,f1,...,fn-1)
    # ATTENTION: Python arrays and lists start at index "0" !!!!
    
    f0 =  y[1]
    f1 = -y[0]
    
    return np.array([f0,f1])
############################################################################
############################################################################
# other relevant data
t_n = [0]; T = 50;        # time interval: t in [t0,T]
y_n = [np.array([1,0])]   # initial condition

n = 10000                  # time interval partition (discretization)
dt = (T-t_n[-1])/n
while t_n[-1] < T:
    y_n.append(y_n[-1] + dt*phi(t_n[-1],y_n[-1],dt,f))
    t_n.append(t_n[-1] + dt)
    
    dt = min(dt, T-t_n[-1])

y_n = np.array(y_n)

# appropriate/adequate plots... look for information in
# https://matplotlib.org/gallery/index.html

#plt.figure(figsize=(15,5))
plt.plot(t_n, y_n[:,0], color='black', linestyle=(0,(1,1,3,1)),
             label = 'y_1(t)  (in y_1 units)')
#plt.plot(t_n, y_n[:,1], c = 'k', label = 'y_2(t)  (in y_2 units)')
plt.xlabel('time t   (in units)')
plt.ylabel('y  state variables')
plt.title('Numerical Approximation of State Variables')
plt.legend()
plt.show()

plt.plot(t_n, y_n[:,1], c = 'k', label = 'y_2(t)  (in y_2 units)')
plt.xlabel('time t   (in units)')
plt.ylabel('y  state variables')
plt.title('Numerical Approximation of State Variables')
plt.legend()
plt.show()

