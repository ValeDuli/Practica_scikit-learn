# El objetivo es clasificar a los estudiantes de una secundaria entre estudiantes con necesidades especiales (2) y estudiantes regulares (1) utilizando como 
# caracteristicas la edad, el grado al que asiste y su calificacion. Esto permitira definir si un alumno debe recibir antencion especial de parte especial de 
# parte de la comunidad escolar y mitigar resagos.

# Librerias
# Manejo de datos y archivos
import pandas as pd

# Manejo de matrices y funciones matematicas
import numpy as np

# Graficacion 
import matplotlib.pyplot as plt

# Algoritmos y herramientas para definicion del cluster
from sklearn.cluster import KMeans

# Importacion de Datos
df=pd.read_csv("scikit-learn/CalifML-2.csv")
print(df)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(df['Edad'],df['Grado'],df['Calificación'],color="blue")
ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificación')
#plt.show()

# Seleccion de caracteristicas para clustering
x=df.iloc[:,0:3].values
print(x)

# Identificacion de Clusters
kmeans=KMeans(n_clusters=2)
kmeans.fit(x)
clustersid=kmeans.fit_predict(x)
print(clustersid)

# Visualizacion en tabla
df["Cluster"]=pd.Series(clustersid+1)
print(df)

# Visualizacion Grafica
print(kmeans.cluster_centers_)
xc=kmeans.cluster_centers_
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
for i in range(0,len(clustersid)):
    if clustersid[i]==0:
        c="blue"
    if clustersid[i]==1:
        c="red"
    ax.scatter(x[i,0],x[i,1],x[i,2],color=c)
ax.scatter(xc[:,0],xc[:,1],xc[:,2],color="green")
ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificación')
#plt.show()

# Comparativa con desempeño original
y=df.iloc[:,4].values
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
for i in range(0,len(y)):
    if y[i]==1:
        c="blue"
    if y[i]==2:
        c="red"
    ax.scatter(x[i,0],x[i,1],x[i,2],color=c)
ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificación')
plt.show()