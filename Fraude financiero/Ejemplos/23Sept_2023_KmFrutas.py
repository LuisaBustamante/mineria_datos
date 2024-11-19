# -*- coding: utf-8 -*-
"""
Created on Tue May  3 20:02:20 2022
Clustering con Frutas_Modelo de Zapata
@author: Fernando Gutierrez
"""
# In[1]:
#importar liberias
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.preprocessing import normalize 
from sklearn.cluster import KMeans
from copy import deepcopy


import warnings
warnings.filterwarnings("ignore")

# In[2]:
''''
- volumen en cent. cubicos
- masa emn gramos y los otros en cms
'''
    
#Distancia euclidea
def distancia(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#Se importnan los datos
fruits  = pd.read_csv('frutas_pro.csv')
X_fruits = fruits[['mass', 'width', 'height', 'color_score', 'volumen']]
y_fruits = fruits[['fruit_label']]

print(X_fruits.head(10))
print(y_fruits)

print("Número de características:", len(fruits.columns))
print("Longitud del conjunto de datos:", len(fruits))
# In[3]:
# Comprobamos si alguna columna tiene valores nulos
fruits.isna().any()  
# In[4]:
print('#### Mezclar la base de datos###############')
#Mezclar la base de datos
fruits = fruits.sample(frac=1).reset_index(drop=True)
fruits.head(10)
# In[5]:
''''ANALISIS EXPLORATORIO DE LOS DATOS

Se hace una Analisis Bivariable'''

#Scatter plot- Visualizar por masa y color
fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(fruits['fruit_name'], fruits['mass'], s=250)

plt.xlabel('fruit_name')
plt.ylabel('mass')

plt.show()

#Visualizar por 'width', 'height
fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(fruits['fruit_name'], fruits['volumen'], s=250)

plt.xlabel('fruit_name')
plt.ylabel('volumen')

plt.show()

#Visualizar por alto y color

fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(fruits['color_score'], fruits['height'], s=250)

plt.xlabel('color_score')
plt.ylabel('height')

plt.show()
  
# In[6]:
''' Escalado y Normalización de los datos
    Nota: 
        - La Normalzación es un Reescalado de caracteristicas [0,1].>Aplica el min-max
        - La Estandarización. Las caracteristicas a una media de una std de 1, 
          toman forma de distribuc.Normal y mantiene info de outliers
    '''
    
    #Estandarización
    from sklearn.preprocessing import StandardScaler
    # Se escalan los datos para que todos los atributos sean comparable
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_fruits)
       
    #Normalzacion
    # from sklearn.preprocessing import MinMaxScaler

    # # Se escalan los datos para que todos los atributos sean comparable
    # X_scaled = MinMaxScaler().fit(X_train).transform(X_train)  

# In[12]:
#Forma No.1 de la Curva de Elbown para Encontrar el No. clusters a armar
Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X_fruits).score(X_fruits) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
# In[13]:
'''Modelo #2 Mej.del Cálculo del número de Clusters (K)'Curva de Elbown')'''

inercias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(X_fruits)    
    inercias.append(kmeans.inertia_)

plt.figure(figsize=(6, 5), dpi=100)
plt.scatter(range(2, 10), inercias, marker="o", s=180, color="purple")
plt.xlabel("Número de Clusters", fontsize=25)
plt.ylabel("Inercia", fontsize=25)
plt.show()
# In[11]:
'''Tercer Metodo codo del libro de Machien Learnig sebatin Rasckka'''
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X_fruits)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
#plt.savefig('images/11_03.png', dpi=300)
plt.show()
    
# In[10]:
'''Inicialización de los cluster centroides donde (K=4)'''
K=2

#Representación grafica de los datos
plt.rcParams['figure.figsize'] = (12,8)

feature1 = X_fruits['volumen'].values
feature2 = [10] * len(feature1)

plt.scatter(feature1, feature2, c='black', s=5)
plt.show()

# In[12]:

#Inicializacion de las coordenadas Y y X para cada cluster centroid
C_x = np.random.randint(0, np.max(feature1), size=K)
C_y =[10] * K

#Representación Grafica
plt.scatter(feature1, feature2, c='black', s=5)
plt.scatter(C_x, C_y, marker='*', c='r', s=600)
plt.show()

# In[13]:
#Inicio del algortimo Kmeans
#Agrupamos los datops en Matrices
X = np.array(list(zip(feature1 , feature2)))
C = np.array(list(zip(C_x , C_y)), dtype=np.float32)

# In[14]:
#Objeto para almacenar el valor de los centroides cuando se actualicen
C_anterior = np.zeros(C.shape)

# Etiquetas de los clusteres
clusters = np.zeros(len(X))

# In[15]:
#Se calcula la distancia entre los nuevos centroides y los anteriores
# se demora un buen rato.......'

dist= distancia(C, C_anterior, None)

while dist !=0:
    #se asigna cada valor al cluster más cercano
    for i in range(len(X)):
        distancias = distancia(X[i], C)
        #se elige la menor
        c_min = np.argmin(distancias)
        clusters[i] = c_min
        #asignación de los nuevos valores a los centroides
        for i in range(K):
            datos_asignados = [X[j] for j in range(len(X)) if clusters[j] == 1]
            # se calcula la media de  los elementos asignados
            C[i] = np.mean(datos_asignados, axis=0)
            #se guardan los valores anteriores
            C_anterior = deepcopy(C)
            #Se compreiba si la posición de los centroides ha variado
            dist = distancia(C, C_anterior, None)

# In[16]:
#representaciín grafica del resualtdo 
COLORES = ['y', 'r', 'g', 'b']
fig, ax = plt.subplots()
for i in range(K):
    datos_asignados = np.array([X[j] for j in range(len(X)) if clusters[j] ==i])
    ax.scatter(datos_asignados[:, 0], datos_asignados[:, 1], s=5, c=COLORES[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=600, c='b')
    plt.show()
    
#Hasta aquillega ese codigo
    
