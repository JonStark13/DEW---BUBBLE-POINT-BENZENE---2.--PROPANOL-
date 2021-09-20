# -*- coding: utf-8 -*-
"""
Created on Fri May 04 16:34:57 2018

@author: Grupo 6 
"""

import numpy
from scipy import optimize
from scipy.constants import R as Rgi # (m3 Pa)/(mol K) Constante de los gases ideales

def Gamma_6(x,T): # Cálculo del coeficiente de Actividad  (NRTL)
    alfha = 0.47
    R = 1.987207 ##cal/mol K
    g = numpy.array([875.0,355.0])
    tao = g/(R*T)
    G  = numpy.exp(-alfha*tao)
    gam1 = numpy.exp((x[1]**2)*((((G[1]/(x[0]+x[1]*G[1]))**2)*tao[1])+(tao[0]*G[0])/((x[1]+x[0]*G[0])**2)))
    gam2 = numpy.exp((x[0]**2)*((((G[0]/(x[1]+x[0]*G[0]))**2)*tao[0])+(tao[1]*G[1])/((x[0]+x[1]*G[1])**2)))   
    return numpy.array([gam1,gam2])

def PsatAntoine_6(i,T): # Psat[Kpa]; T[°C]
#Ecuación de Antoine. 
    A = [13.7819, 16.6796]
    B = [2726.81, 3640.20]
    C = [217.572, 219.610]
    return numpy.exp( A[i] - B[i]/( (T-273.15) + C[i] ))*1000 ##Pa
    
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

def P_Rocio(y,T):
    numero_s = len(y)
    PHI = numpy.array([1.0 for i in range(0,numero_s)])
    gamma = numpy.array([1.0 for i in range(0,numero_s)])
    Psat = numpy.array([PsatAntoine_6(i,T) for i in range(0,numero_s)])
    P = 1.0/sum(y*PHI/(gamma*Psat))
    x = y*PHI*P/(gamma*Psat) 
    gamma = Gamma_6(x,T)
    #print type(y),type(PHI),type(gamma),type(Psat)
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

def P_Burbuja(x,T):
    numero_s  = len(x)
    PHI = numpy.array([1.0 for i in range(0,numero_s )])
    gamma = numpy.array([1.0 for i in range(0,numero_s )])
    Psat = [PsatAntoine_6(i,T) for i in range(0,numero_s )]
    P= sum(x*gamma*Psat/(PHI))
    y= x*gamma*Psat/(PHI*P)
    gamma = Gamma_6(x,T)
    P= sum(x*PHI*gamma*Psat)
    ErrorP = 0.1
    ErrorPermitidoP = 1.0E-7
    while ErrorP >= ErrorPermitidoP:
        P_E0 = P
        PHI = numpy.array([Phi_6(y,T,P_E0)[i]/Phi_6(y,T,Psat[i])[i] for i in range(0,numero_s )])
        y= x*gamma*Psat/(PHI*P)
        P= sum(x*gamma*Psat/(PHI))
        ErrorP = abs(P_E0 - P)
    return y,P


##FLASH
def flash(z,T,P):
    numero_s = len(z)
    Procio = P_Rocio(z,T)[1]
    Pburbu = P_Burbuja(z,T)[1]
    if P<Procio  or P>Pburbu:
        print ("ERROR NO HAY SEPARACIÓN")
    Nu = (Pburbu-P)/( Pburbu - Procio) ##v/f
    PHI = Phi_6(z,T,P)/numpy.array([Phi_6(z,T,PsatAntoine_6(i,T))[i] for i in range (0,numero_s)])
    gamma= Gamma_6(z,T)
    Psat =[PsatAntoine_6(i,T) for i in range(0,numero_s)]
    "CALCULOS"
    x=z
    y=z
    errorx = numpy.array([0.1 for i in range (0,numero_s)])
    errory = numpy.array([0.1 for i in range (0,numero_s)])
    errorNu = 0.1 #Error inicial
    ErrorPermitido = numpy.array([1.0E-10 for i in range (0,numero_s)])
    while (errorx>=ErrorPermitido).all() or (errory>=ErrorPermitido).all() or errorNu>=ErrorPermitido[0]:
        x_E0 = x
        y_E0 = y
        Nu_E0 = Nu
        k = gamma*Psat/(PHI*P) #K=x/y
        def FO(Nu):
            F = sum(z*(k-1)/(1+Nu*(k-1)))
            dFdNu = -sum(z*(k-1)**2/(1+Nu*(k-1))**2)
            return F + dFdNu*(Nu-Nu_E0)
        
        x = z/(1+Nu*(k-1))
        y = x*k
        x = x/sum(x)
        y = y/sum(y)
        Nu = optimize.fsolve(FO,Nu)[0]
        gamma = Gamma_6(x,T)
        PHI = Phi_6(y,T,P)/numpy.array([Phi_6(y,T,Psat[i])[i] for i in range(0,numero_s)])
        errorx = abs(x_E0 - x)
        errory = abs(y_E0 - y)
        errorNu = abs(Nu_E0 - Nu)
    return x,y,Nu            

    
#==============================================================================
# Interfaz
#==============================================================================
print ("                   Parra Arias David Fernando               ")
print ("")
print ("            CÁLCULO DEL FLASH ISOTÉRMICO  PARA LA           ")
print ("             MEZCLA BINARIA BENCENO - 2-PROPANOL            ")
print ("")
T = 74 + 273.15 #K 
P = 101058 #Pa 
z = numpy.array([0.3,0.7])
print (" fase liquida\t fase vapor")
print ("x   \t                  y\t        Nu")
for i in range(0,len(z)):
    [x,y,Nu] = flash(z,T,P)
    if i == 0:    
        print ("{0:.4f}\t               {1:.4f}\t     {2:.7f}".format(x[0],y[0],Nu))
    else :
        print ("{0:.4f}\t               {1:.4f}\t     {2:.7f}".format(x[1],y[1],Nu))
