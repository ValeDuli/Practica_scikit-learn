# El objetivo es clasificar a los estudiantes de una secundaria entre estudiantes con necesidades especiales (2) y estudiantes regulares (1) utilizando 
# como caracteristicas la edad, el grado al que asiste y su clasificacion. Esto permitira definir si un alumno debe recibir atencion especial de parte 
# de la comunidad escolar y mitigar resagos.

# Librerias
# Manejo de Datos
import pandas as pd

# Manejo de matrices y funciones matematicas
import numpy as np

# Graficacion
import matplotlib.pyplot as plt

# Algoritmo de apoyo para clasificacion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Ingresar datos y generar dataframes.
# Las clasificaciones ya las conocemos a apriori, por lo cual se tienen los datos de necesidades especiales (2) y alumnos regulares (1) en desempeño.
# Esto nos dara un criterio matematico para determinar, considerando los datos base (Edad, Grado y Calificacion), es potencial candidato para atencion especial.
df=pd.read_csv("CalifML.csv")
print(df)

# Agrupacion de datos
x=df.iloc[:,0:3].values
y=df.iloc[:,3].values

# Escalamiento de valores y entrenamiento
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.4, random_state=0)
print(xtest)
print(ytest)
xg=xtest
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)

# Parametros de algortimo LDA
lda=LDA(n_components=1)
xtrain=lda.fit_transform(xtrain,ytrain)

# Ejecucion de la clasificacion
print(lda.predict(xtest))
yclass=lda.predict(xtest)
print(ytest)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
for i in range (0, len(ytest)):
    if yclass[i]==1:
        c="blue"
    if yclass[i]==2:
        c="red"
    ax.scatter(xg[i,0],xg[i,1],xg[i,2],color=c)
    
ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificacion')


fig=plt.figure()
ax=fig.add_subplot(projection='3d')
for i in range(0,len(ytest)):
    if ytest[i]==1:
        c="blue"
    if ytest[i]==2:
        c="red"
    ax.scatter(xg[i,0],xg[i,1],xg[i,2],color=c)
    
ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificacion')
plt.show()