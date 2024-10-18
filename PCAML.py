# El objetivo es clasificar a los estudiantes de una secundaria entre estudiantes con necesidades especiales (2) y estudiantes regulares (1) utilizando como 
# caracteristicas la edad, el grado al que asiste y su calificacion. Esto permitira definir si un alumno debe recibir atencion especial de parte de la comunidad 
# escolar y mitigar resagos.
# Para el caso de PCA, la reduccion dimensional nos permite pasar de una representacion tridimensional a una bidimensional donde los componentes esnta asociados 
# a la varianza de los datos originales.

# Librerias
# Manejo de datos y archivos
import pandas as pd

# Manejo de matrices y funciones matematicas
import numpy as np

# Graficacion
import matplotlib.pyplot as plt

# Algotitmos y herramientas para definicion de cluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Importacion de Datos
df=pd.read_csv("scikit-learn/CalifML-3.csv")
print(df)

# Seleccion de caracteristicas para clustering
x=df.iloc[:,0:3].values
print(x)

std=StandardScaler(with_mean=True, with_std=True)
xst=std.fit_transform(x)
print(xst)

# Ejecucion de algoritmo
pca=PCA(n_components=2)
xpca=pca.fit_transform(xst)
print(xpca)

# Efectividad del PCA
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

# Graficacion
plt.scatter(xpca[:,0],xpca[:,1])
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()
