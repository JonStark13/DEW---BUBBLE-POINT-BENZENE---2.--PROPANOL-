# -*- coding: utf-8 -*-
"""
Created on Wed May 02 17:14:40 2018

@author: hp ProBook
"""

import numpy
from scipy.constants import R as Rgi # (m3 Pa)/(mol K) Constante de los gases ideales
import matplotlib.pyplot as plt
#==============================================================================
# CALCULO P-ROCIO 
#==============================================================================
def PsatAntoine_6(i,T): # Psat[Kpa]; T[°C]
#Ecuación de Antoine. 
    A = [13.7819, 16.6796]
    B = [2726.81, 3640.20]
    C = [217.572, 219.610]
    return numpy.exp( A[i] - B[i]/( (T-273.15) + C[i] ))*1000 ##Pa

#Cálculo de Fugacidad 
def Phi_6(y,T,P): #Ecuación Peng-Robinson (Benceno, 2-Propanol)    
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
    ## Método roots
    a_raiz = 1.0
    b_raiz = (deltaprima-Bprima-1.0)
    c_raiz = thetaprima+epprima-deltaprima*(Bprima+1.0)
    d_raiz = -(epprima*(Bprima+1.0)+thetaprima*nprima)
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
    
def Gamma_6(x,T): # Cálculo del coeficiente de Actividad  (NRTL)
    alfha = 0.47
    R = 1.987207 ##cal/mol K
    g = numpy.array([875.0,355.0])
    tao = g/(R*T)
    G  = numpy.exp(-alfha*tao)
    gamma =numpy.array([numpy.exp((x[1]**2)*((((G[1]/(x[0]+x[1]*G[1]))**2)*tao[1])+(tao[0]*G[0])/((x[1]+x[0]*G[0])**2))),numpy.exp((x[0]**2)*((((G[0]/(x[1]+x[0]*G[0]))**2)*tao[0])+(tao[1]*G[1])/((x[0]+x[1]*G[1])**2)))])    
    return gamma
    
  
 
def P_Rocio(y,T):
    numero_s = len(y)
    PHI = numpy.array([1.0 for i in range(0,numero_s)])
    gamma = numpy.array([1.0 for i in range(0,numero_s)])
    Psat = [PsatAntoine_6(i,T) for i in range(0,numero_s)]
    P = 1.0/sum(y*PHI/(gamma*Psat))
    x = y*PHI*P/(gamma*Psat) 
    gamma = Gamma_6(x,T)
    P = 1.0/sum(y*PHI/(gamma*Psat))
    ErrorP = 0.1
    ErrorPermitidoP = 1.0E-7
    ErrorPermitidogamma = numpy.array([1.0E-12 for i in range(0,numero_s)])
    while ErrorP >= ErrorPermitidoP:
        P_E0 = P
        PHI = numpy.array([Phi_6(y,T,P_E0)[i]/Phi_6(y,T,Psat[i])[i] for i in range(0,numero_s)])
        Errorgamma = numpy.array([0.1 for i in range(0,numero_s)]) 
        while (Errorgamma>=ErrorPermitidogamma).all():
            gamma_E0 = gamma
            x = y*PHI*P_E0/(gamma_E0*Psat)
            x = x/sum(x)
            gamma = Gamma_6(x,T)
            Errorgamma = abs(gamma_E0 - gamma)
        P = 1.0/sum(y*PHI/(gamma*Psat))
        ErrorP = abs(P_E0 - P)
    return x,P

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
P_graf = []
T = 344.63 #K

print ("y0\tx0\tP(Pa)")
for y0 in numpy.arange(0,1+0.02,0.02):
    y = numpy.array([y0,1-y0])+1E-10
    [x,P] = P_Rocio(y,T)
    x_graf = x_graf + [x[0]]
    y_graf = y_graf + [y[0]]
    P_graf = P_graf + [P]
    print ("{0:.4f}\t{1:.4f}\t{2:.7f}".format(y[0],x[0],P))

# Grafica Presion de Rocio
plt.plot(x_graf,P_graf, color='green', linestyle='-', marker='', label='Linea de presion  de burbuja')
plt.plot(y_graf,P_graf, color='orange', linestyle='-', marker='', label='Linea de  presion de rocio')
plt.legend(loc=0)
plt.title('DIAGRAMA  PRESION DE ROCIO  ', color='k', size='20') 
plt.xlabel(" x-y [BENCENO]", color='k', size='16') 
plt.ylabel("PRESION (Pa)", color='k', size='16') 
plt.xlim((0,1))
plt.show(True)

#Grafica  X-Y 
plt.plot(x_graf,y_graf, color='black', linestyle='-', marker='', label=' x ')
plt.plot(x_graf,x_graf, color='blue', linestyle='-', marker='', label='y')
plt.legend(loc=0)
plt.title('DIAGRAMA X-Y ', color='k', size='20') 
plt.xlabel("X ", color='k', size='16') 
plt.ylabel("Y ", color='k', size='16') 
plt.xlim((0,1))
plt.ylim((0,1))
plt.show(True)