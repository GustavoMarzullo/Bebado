import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from numba import jit


#Bêbado 2.0
@jit
def direcao():
    return np.array(random.choice([(1,0),(-1,0),(0,1),(0,-1)]))

@jit
def direcao_euclides():
    x=random.uniform(-1, 1)
    y=random.uniform(-1, 1)
    return np.array([x,y])

@jit
def caminhar(n,xmax,ymax,tipo=1):
    '''Caminha n passos aleatórios em um campo retangular definido por +/-xmax e +/-ymax.'''
    origem=np.array((0.0,0.0))
    if tipo==1:
        for i in range(n):
            origem+=direcao()
            if origem[0]>xmax:
                origem[0]=xmax
            if origem[0]<-xmax:
                origem[0]=-xmax
            if origem[1]>ymax:
                origem[1]=ymax
            if origem[1]<-ymax:
                origem[1]=-ymax
        return origem
    else:
        for i in range(n):
            origem+=direcao_euclides()
            if origem[0]>xmax:
                origem[0]=xmax
            if origem[0]<-xmax:
                origem[0]=-xmax
            if origem[1]>ymax:
                origem[1]=ymax
            if origem[1]<-ymax:
                origem[1]=-ymax
        return origem

@jit
def dist(P):
    '''Retorna a distância do ponto P(x,y) até a origem.'''
    return (P[0]**2+P[1]**2)**0.5

@jit
def distmedia(n,x,xmax,ymax,tipo):
    '''Retorna a distância média e o desvio-padrão de x séres de n passos.'''
    distancias=[]
    for i in range(x):
        distancias.append(dist(caminhar(n,xmax,ymax,tipo)))
    return stats.tmean(distancias),stats.tstd(distancias)

@jit
def grafico(n,step,x,xmax,ymax,tipo=1):
    '''Faz um gráfico de 'número de passos'X'distância média percorrida'.\nn=> número de passos final.\
\nstep=> passos para ir de 0 até n. Ex: n=1 será 1,2,3,4,5... e n=2 será 1,3,5,7...\
\nx=> número de séries para cada passo.\
\nxmax e ymax=> delimitadores do campo no qual o bêbado anda.'''
    medias=[]
    n_lista=[]
    i=0
    while i<n:
        medias.append(distmedia(i,x,xmax,ymax,tipo)[0])
        n_lista.append(i)
        i+=step
    plt.figure(1)
    plt.plot(n_lista,medias,color='grey', linestyle='dashed', marker='o',markerfacecolor='black', markersize=4)
    plt.title('Bêbado')
    plt.xlabel('n')
    plt.ylabel('Distância')
    plt.show()
    #return medias,n_lista

@jit
def scatter(n,x,xmax,ymax,tipo=2):
    '''Faz uma gráfico de dispersão dos locais onde o bêbado parou para x séries.'''
    posx,posy=[],[]
    for i in range(x):
        pos=caminhar(n,xmax,ymax,tipo)
        posx.append(pos[0])
        posy.append(pos[1])
    plt.scatter(posx,posy,color='black',marker='.')
    plt.xlim(-xmax,xmax)
    plt.ylim(-ymax,ymax)
    plt.title(str(x)+' séries de '+str(n)+' passos.')
