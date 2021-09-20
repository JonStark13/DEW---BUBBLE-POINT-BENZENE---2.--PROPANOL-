# -*- coding: utf-8 -*-
"""
Created on Wed May 02 17:48:19 2018

@author: hp ProBook
"""

import numpy
from scipy.constants import R as Rgi # (m3 Pa)/(mol K) Constante de los gases ideales
from scipy import optimize #fsolve, fmin
import matplotlib.pyplot as plt
#==============================================================================
# CÁLCULO T-BURBUJA
#=======================================================================

#Datos de referencia
Tamb = 25 + 273.15 #K

def PsatAntoine_GR3(i,T): # Psat[Kpa]; T[°C]
    A = [13.7819, 16.6796]
    B = [2726.81, 3640.20]
    C = [217.572, 219.610]
    return numpy.exp( A[i] - B[i]/( (T-273.15) + C[i] ))*1000 ##Pa

#Cálculo de Fugacidad 
def Phi_GR3(y,T,P): #Ecuación Peng-Robinson (Benceno, 2-Propanol)     
    numero_s = len(y)
    Pc = numpy.array([4895000.0,4764000.0])    #Pa  
    Tc = numpy.array([562.05,508.3])           #K  
    Tr = (T/Tc)
    w = numpy.array([0.212,0.665])
    k = numpy.array([[0.0,0.01349],[0.01349,0.0]])
    ##Bprima 
    b = (0.0778*Rgi*Tc)/Pc
    bm = sum(y*b)
    Bprima = (bm*P)/(Rgi*T)
    ##Delta Prima  
    delta = 2.0*bm
    deltaprima = (delta*P)/(Rgi*T)
    ## Theta prima
    alfa = [(1.0+(0.37464 + 1.54226*w[i] - 0.2699*w[i]**2)*(1.0 - numpy.sqrt(Tr[i])))**2 for i in range(0,numero_s)]
    a = (0.45724*(Rgi*Tc)**2)/Pc
    aap = (a*alfa)
    aalfaij = numpy.array([[(aap[i]*aap[j])**(0.5)*(1.0-k[i][j]) for i in range(0,numero_s)] for j in range(0,numero_s)])
    aalfam = sum(sum(numpy.array([[y[i]*y[j] for i in range(0,numero_s)] for j in range(0,numero_s)])*aalfaij))
    thetaprima = (aalfam*P)/((Rgi*T)**2)
    ##Epsinol prima
    epsinol = -(bm**2)
    epprima = epsinol*((P/(Rgi*T))**2)
    ##n prima
    nprima = (bm*P)/(Rgi*T)
    ##Cardano
    a_raiz = 1.0
    b_raiz = (deltaprima-Bprima-1.0)
    c_raiz = thetaprima + epprima-deltaprima*(Bprima+1.0)
    d_raiz = -(epprima*(Bprima+1.0)+ thetaprima *nprima)
    r = numpy.roots([a_raiz,b_raiz,c_raiz,d_raiz])
    Z = max(r.real[abs(r.imag)<1e-5])
    
    ## Cálculo Fugacidad
    AA = (2.0/aalfam)*sum(y*aalfaij)
    A = (aalfam*P)/((Rgi**2)*(T**2))
    B = (bm*P)/(Rgi*T)
    BB = b/bm
    ln = (BB*(Z-1.0))-numpy.log(Z-B)-((A/(2.0*numpy.sqrt(2)*B))*(AA-BB)*numpy.log((Z+(numpy.sqrt(2.0)+1.0)*B)/(Z-(numpy.sqrt(2.0)-1.0)*B)))
    phi = numpy.exp(ln) 
    return phi
    
def Gamma_GR3(x,T): # Cálculo del coeficiente de Actividad  (NRTL)
    alfha = 0.47
    R = 1.987207 ##cal/mol K
    g = numpy.array([875.0,355.0])
    tao = g/(R*T)
    G  = numpy.exp(-alfha*tao)
    gamma =numpy.array([numpy.exp((x[1]**2)*((((G[1]/(x[0]+x[1]*G[1]))**2)*tao[1])+(tao[0]*G[0])/((x[1]+x[0]*G[0])**2))),numpy.exp((x[0]**2)*((((G[0]/(x[1]+x[0]*G[0]))**2)*tao[0])+(tao[1]*G[1])/((x[0]+x[1]*G[1])**2)))])    
    return gamma

def T_Burbuja(x,P):
    numero_s = len(x)
    PHI = numpy.array([1.0 for i in range(0,numero_s)])
    def FO_Tsat(T,P,i):
        return P - PsatAntoine_GR3(i,T)
    Tsat = numpy.array([optimize.fsolve(lambda T:FO_Tsat(T,P,i),Tamb)[0] for i in range(0,numero_s)])
    T = sum(x*Tsat)
    Psat = [PsatAntoine_GR3(i,T) for i in range(0,numero_s)]
    gamma = Gamma_GR3(x,T)
    Psatj = P/sum((x*gamma/PHI)*(Psat/Psat[0]))
    T = optimize.fsolve(lambda T:FO_Tsat(T,P,0),Tamb)[0]
    ErrorT = 0.1
    ErrorPermitidoT = 1.0E-10
    while ErrorT >= ErrorPermitidoT:
        T_E0 = T
        Psat = [PsatAntoine_GR3(i,T_E0) for i in range(0,numero_s)]
        y = x*gamma*Psat/(PHI*P)
        y = y/sum(y)
        PHI = numpy.array([Phi_GR3(y,T_E0,P)[i]/Phi_GR3(y,T_E0,Psat[i])[i] for i in range(0,numero_s)])
        gamma = Gamma_GR3(x,T_E0)
        Psatj = P/sum((x*gamma/PHI)*(Psat/Psat[0]))
        def FO_T_E1(T,i):
            return Psatj - PsatAntoine_GR3(i,T)
        T = optimize.fsolve(lambda T:FO_T_E1(T,0),T_E0)[0]
        ErrorT = abs(T_E0 - T)
    return y,T

#==============================================================================
# Interfaz
#==============================================================================
print ("                   Parra Arias David Fernando               ")
print ("")
print ("            CÁLCULO DEL FLASH ISOTÉRMICO  PARA LA           ")
print ("             MEZCLA BINARIA BENCENO - 2-PROPANOL            ")
print ("")
x_graf = []
y_graf = []
T_graf = []
P = 101058 #Pa

print ("y0\          tx0       \    tT(k)")
for x0 in numpy.arange(0,1+0.02,0.02):
    x = numpy.array([x0,1-x0])+1E-10
    [y,T] = T_Burbuja(x,P)
    x_graf = x_graf + [x[0]]
    y_graf = y_graf + [y[0]]
    T_graf = T_graf + [T]
    print ("{0:.4f}\t{1:.4f}\t{2:.7f}".format(y[0],x[0],T))

#Grafica GRAFICA DE T-BURBUJA
plt.plot(x_graf,T_graf, color='blue', linestyle='-', marker='', label='Linea de rocio')
plt.plot(y_graf,T_graf, color='green', linestyle='-', marker='', label='Linea de burbuja')
plt.legend(loc=0)
plt.title('DIAGRAMA TEMPERATURA BURBUJA ', color='k', size='20') 
plt.xlabel("x-y [BENCENO] ", color='k', size='16') 
plt.ylabel("TEMPERATURA (K)", color='k', size='16') 
plt.xlim((0,1))
plt.show(True)

#Grafica GRAFICA X-Y
plt.plot(x_graf,y_graf, color='black', linestyle='-', marker='', label='')
plt.plot(x_graf,x_graf, color='blue', linestyle='-', marker='', label='')
plt.legend(loc=0)
plt.title('DIAGRAMA X-Y ', color='k', size='20') 
plt.xlabel("X", color='k', size='16') 
plt.ylabel("Y", color='k', size='16') 
plt.xlim((0,1))
plt.ylim((0,1))
plt.show(True)