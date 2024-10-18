#Ejemplo adaptado de
#Chapra y Canale (2006). Metodos Numericos para ingenieros. 5a Edicion. Mc Graw Hill.
#La poblacion (p) de una comunidad pequeña en los suborbios de uina ciudad crece con rapidez durante un periodo de 20 años en la siguente forma:

# t          0   3   5   8   10  15  20
# p         100 150 200 550 650 950 2000
# control    0   0   1   0   1   0   0

# La serie de control indica los años en que se implementaron campañas de planificacion familiar
# consoderando que se requiere frenar la tasa de natalidad y evitar una explosion demografica.

#Pronosicar la poblacion entre 5 años y analice el patron de insercion de campañas de planificacion familiar

#Ejemplo lineal

# Librerias a utilizar

#Datos y dataframes
import pandas as pd

#Algoritmos de regresion
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Graficacion y manejo de matrices
import matplotlib.pyplot as plt
import numpy as np

#Ingrsar datos y generar data frame
df=pd.read_csv("DataReg.csv")
print(df)

#Generar grafico
df.plot.scatter(x='Tiempo',y='Poblacion')

#Entrenamiento y prediccion
regl=LinearRegression()
regl.fit(df[['Tiempo']],df['Poblacion'])

#Preduccion para la poblacion en 5 años (año 25)
print(regl.predict([[25]]))

df["modelolin"]=pd.Series(regl.predict(df[['Tiempo']]))
print(df)

#Ecuacion de la recta generada y=mx+b
print("p =",regl.coef_,"*t",regl.intercept_)

#Prediccion sobre el modelo
print(regl.coef_*df['Tiempo']+regl.intercept_)

#Graficacion
ax=df.plot.scatter(x='Tiempo',y='Poblacion')
df.plot.line(x='Tiempo',y='modelolin',ax=ax,color="red")

#Regresion Polinomial
polyreg=PolynomialFeatures(degree=2) #Grado del polinomio
#Definicion de variables
x_poli=polyreg.fit_transform(df[['Tiempo']])
print(x_poli)

#Solucion del modelo
regcuad=LinearRegression()
regcuad.fit(x_poli,df['Poblacion'])

#Prediccion a 5 años(Año 25)
print(regcuad.predict(polyreg.fit_transform([[25]])))

#Evaluacion del resto de valores
df['modelocuad']=pd.Series(regcuad.predict(x_poli))
print(df)

#Modelo generado
print(regcuad.coef_[0]+(regcuad.coef_[1]*df['Tiempo'])+(regcuad.coef_[2]*df['Tiempo']**2)+regcuad.intercept_)

print("p =",regcuad.coef_[2],"* t^2 + ",regcuad.coef_[1],"*t + ",regcuad.coef_[0]+regcuad.intercept_)

#Graficacion
ax=df.plot.scatter(x='Tiempo',y='Poblacion')
df.plot.line(x='Tiempo',y='modelocuad',ax=ax, color="red")

print("Correlacion para modelo lineal ", r2_score(df['Poblacion'],df['modelolin']))
print("Correlacion para modelo polinomial ", r2_score(df['Poblacion'],df['modelocuad']))

#Regresion Logistica: Seleccion de datos
xlogis=df.iloc[:,[0,1]].values
ylogis=df.iloc[:,2].values

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(xlogis[:,0],xlogis[:,1],ylogis)
ax.set_xlabel('Tiempo')
ax.set_ylabel('Poblacion')
ax.set_zlabel('Control')
plt.show()

#Escalamiento de valores de variables "continuas"
xstan=StandardScaler()
xtr=xstan.fit_transform(xlogis)
#print(xtr)

#Generacion del modelo
rlog=LogisticRegression()
rlog.fit(xtr,ylogis)
df['modelolog']=pd.Series(rlog.predict(xtr))
print(df)

#Parametros del modelo
print(f"b: {rlog.intercept_}")
print(f"w: {rlog.coef_}")