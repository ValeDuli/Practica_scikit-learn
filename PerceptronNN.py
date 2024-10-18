# El objetivo es clasificar a los estudiantes de una secundaria entre estudiantes con necesidades especiales (2) y estudiantes regulares (1) utilizando como 
# caracteristicas la edad, eñ grado al que asiste y su calificacion. Esto permitira definir si un alumno debe recibir atencion especial de parte de la comunidad 
# escolar y mitigar resagos.

# Librerias
# Manejo de datos
import pandas as pd

# Manejo de matrices y funciones matematicas
import numpy as np

# Graficacion
import matplotlib.pyplot as plt

# Algoritmo para implementar el perceptron y la red neuronal
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ingresar datos y generar dataframe
# Las clasificaciones ya las conocemos a priori, por lo cual se tienen los datos de necesidades especiales (2) y alumnos regulares (1) en desempeño.
# Esto nos dara un criterio matematico para determinar, considerando los datos base (Edad, Grado y Calificacion), es potencial candidato para atencion especial.
df=pd.read_csv("scikit-learn/CalifML-4.csv")
print(df)

# Agrupacion de datos
x=df.iloc[:,0:3].values
y=df.iloc[:,3].values
#print(x,y)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
for i in range (0,len(y)):
    if y[i]==1:
        c="blue"
    if y[i]==2:
        c="red"
    ax.scatter(x[i,0],x[i,1],x[i,2],color=c)

ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificacion')
#plt.show()

#Implementacion del Perceptron
pcp=Perceptron()
pcp.fit(x,y)
yp=pcp.predict(x)
df["modeloperc"]=pd.Series(yp)
print(df)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
for i in range (0,len(yp)):
    if yp[i]==1:
        c="blue"
    if yp[i]==2:
        c="red"
    ax.scatter(x[i,0],x[i,1],x[i,2],color=c)

ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificacion')
#plt.show()

#Implementacion de la red neuronal
std=StandardScaler()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, random_state=0)
xtrain=std.fit_transform(xtrain)
xtest=std.fit_transform(xtest)
xu=std.fit_transform(x)
NNA=MLPClassifier(random_state=1, max_iter=300).fit(xtrain,ytrain)
ypnn=NNA.predict(xu)
df['modelonn']=pd.Series(ypnn)
print(df)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
for i in range (0,len(ypnn)):
    if ypnn[i]==1:
        c="blue"
    if ypnn[i]==2:
        c="red"
    ax.scatter(x[i,0],x[i,1],x[i,2],color=c)

ax.set_xlabel('Edad')
ax.set_ylabel('Grado')
ax.set_zlabel('Calificacion')
plt.show()