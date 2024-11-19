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

#Visualizar por 'fruta y volumen
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
K=4

#Representación grafica de los datos
plt.rcParams['figure.figsize'] = (12,8)

feature1 = X_fruits['mass'].values
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
#Agrupamos los datos en Matrices
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
    
fruits_2D = fruits[['height', 'mass']]
fruits_2D.sample(5)

print(fruits_2D)

fruits_2D = np.array(fruits_2D)

# In[17]:
#El modelo (K=5)
#Es posible que muestre un cluster vacio....
kmeans_model_2D = KMeans(n_clusters=4, max_iter=300).fit(fruits_2D)

kmeans_model_2D.labels_

#centroides
centroids_2D = kmeans_model_2D.cluster_centers_

print(centroids_2D)

#Graficar Centroides
fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(centroids_2D[:,0], centroids_2D[:,1], c='r', s=250, marker='s')
for i in range(len(centroids_2D)):
    plt.annotate(i, (centroids_2D[i][0], centroids_2D[i][1]), fontsize=30)
#-------------------------------------------
fruits_labels = fruits['fruit_label']
#---------------------------------------

# In[18]:
#visualizar solo las filas únicas de estas dos columnas
df_unique = fruits.drop_duplicates(subset=["fruit_label", "fruit_name"])
print(df_unique)

# In[19]:

colors = ['yellow','blue','green', 'black']

plt.figure(figsize=(12, 8))

plt.scatter(fruits['height'], fruits['mass'], c=fruits['fruit_label'], s=200,
            cmap=matplotlib.colors.ListedColormap(colors), alpha=0.5)

plt.scatter(centroids_2D[:,0], centroids_2D[:,1], c='r', s=250, marker='s')

for i in range(len(centroids_2D)):
    plt.annotate( i, (centroids_2D[i][0], centroids_2D[i][1]), fontsize=30)



#####9'

####Nota: En este caso los centroides no se ajustan, mejorar esto con Kmeans.

#Otra caso las Mediicones estan muy regulares hay que mejoar esto.
print(fruits.head(5))

fruits_features = fruits.drop(['fruit_label','fruit_name','fruit_subtype'],  axis=1)P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''
fruits_features.head()

fruits_labels = fruits['fruit_label']''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P'P''P''P''P''P''P''P''P'P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P'9'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
fruits_labels.sample(10)'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''9''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''9'P''P''P''P''P''P''P''P''P''PP''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P'''P'''P''P''P''P''P''P''P''P''P''P''P''P''P''P''P''


#Ojo falta Normalizar esta data.........


#Entrenamiento del Modelo
kmeans_model = KMeans(n_clusters=8).fit(fruits_features)
kmeans_model.labels_

kmeans_model.cluster_centers_


