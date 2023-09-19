import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # Normalizar
from sklearn.cluster import KMeans
import seaborn as sns
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D

# Leemos el archivo csv del anexoA y anexoB
#dataA= pd.read_csv("anexoA.csv")
dataB = pd.read_csv("anexoB.csv")

# Seleccionamos las columnas 'Abonados', 'DPI' y 'FPI' del DataFrame dataAy dataB
#datosA = dataA[['Abonados', 'DPI', 'FPI']]
datosB = dataB[['Abonados', 'DPI', 'FPI']]

#datosA_i = dataA[['Empresa','Circuito']]
datosB_i = dataB[['Empresa','Circuito']]
# print(dataA.describe())

#  Calcula los valores mínimos y máximos para que estén en el rango [0, 1]
#escaladorA = MinMaxScaler().fit(datosA.values)
escaladorB = MinMaxScaler().fit(datosB.values)

# Datos normalizados 
# Se aplica la transformación de escala a los datos utilizando el escalador previamente ajustado
#datos_normA = pd.DataFrame(escaladorA.transform(datosA.values), columns = ['Abonados', 'DPI', 'FPI'])
datos_normB = pd.DataFrame(escaladorB.transform(datosB.values), columns = ['Abonados', 'DPI', 'FPI'])


# Graficas para Anexo A
datosA.drop([],1).hist()
datos_normA.drop([],1).hist()
plt.title('Histogramas del Anexo A')
sb.pairplot(datos_normA.dropna(), hue='Abonados', height=2,vars=[ 'DPI', 'FPI'], kind='scatter')
plt.show()

# Graficacion para observar el codo para el anexo A
inerciasA = []
for k in range(2, 40):
    # Generación de modelo
    kmeansA = KMeans(n_clusters=k).fit(datos_normA.values)    
    inerciasA.append(kmeansA.inertia_)
#print(inerciasA)

# Codigo para imprimir la gráfica de codo del AnexoA
plt.figure(figsize=(6, 5), dpi=120)
plt.scatter(range(2, 40), inerciasA, marker="o", s=10, color="blue")
plt.xlabel("Número de Clusters", fontsize=10)   # Etiquetas para los ejes x e y
plt.ylabel("Inercia", fontsize=10)
plt.xticks(range(2, 41, 2)) # Definir los ticks (marcas) en el eje x
# Definir los ticks (marcas) en el eje y con valores específicos
yticks = inerciasA[0:9] + inerciasA[11:12]+ inerciasA[19:20] + inerciasA[29:30] + inerciasA[37:38]
plt.yticks(yticks)
plt.yticks(fontsize=5.5)    # Ajustar el tamaño de las etiquetas en los ejes x e y
plt.xticks(fontsize=8)
plt.grid(True, alpha=0.2) # Agregar una cuadrícula al gráfico con transparencia
plt.show() # Mostrar el gráfico

# En base a lo obtenido en el metodo del codo se elige probar varios valores de k
modeloA_k1 = 4

# Generar modelo para el AnexoA con cierto valor de k
kmeansA1 = KMeans (n_clusters= modeloA_k1).fit(datos_normA.values)
labels = kmeansA1.labels_
labels = kmeansA1.predict(datos_normA.values)

datos_normA1 = datos_normA
datos_normA1["cluster"]= kmeansA1.labels_ # Nueva columna con los grupos
A_k1 = pd.concat([datosA_i, datos_normA1], axis=1)  # Uniendo colunmas
#print(A_k1) # Imprimiendo tabla completa

print("Imprimiento los centroides para k=", modeloA_k1 )
print("Centroides:\n", kmeansA1.cluster_centers_)
print("Inercia:", kmeansA1.inertia_)

# Plot the data points and their cluster assignments
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(datos_normA.values[:, 0], datos_normA.values[:, 1], datos_normA.values[:, 2], c=labels, cmap='viridis')
ax.scatter(kmeansA1.cluster_centers_[:, 0], kmeansA1.cluster_centers_[:, 1], kmeansA1.cluster_centers_[:, 2], marker='x', color='red', s=100 , linewidths=3)
ax.set_title("K-means Clustering")
ax.set_xlabel('Abonados')
ax.set_ylabel('DPI')
ax.set_zlabel('FPI')
plt.show()



###########################################################################################################

# Graficas para Anexo B
datosB.drop([],1).hist()
datos_normB.drop([],1).hist()
plt.title('Histogramas del Anexo B')
sb.pairplot(datos_normB.dropna(), hue='Abonados', height=2,vars=[ 'DPI', 'FPI'], kind='scatter')
plt.show()

# Graficacion para observar el codo para el anexo B
inerciasB = []
for k in range(2, 40):
    # Generación de modelo
    kmeansB = KMeans(n_clusters=k).fit(datos_normB.values)    
    inerciasB.append(kmeansB.inertia_)
#print(inerciasB)

# Codigo para imprimir la gráfica de codo del AnexoA
plt.figure(figsize=(6, 5), dpi=120)
plt.scatter(range(2, 40), inerciasB, marker="o", s=10, color="blue")
plt.xlabel("Número de Clusters", fontsize=10)   # Etiquetas para los ejes x e y
plt.ylabel("Inercia", fontsize=10)
plt.xticks(range(2, 41, 2)) # Definir los ticks (marcas) en el eje x
# Definir los ticks (marcas) en el eje y con valores específicos
yticks = inerciasB[0:9] + inerciasB[11:12]+ inerciasB[19:20] + inerciasB[29:30] + inerciasB[37:38]
plt.yticks(yticks)
plt.yticks(fontsize=5.5)    # Ajustar el tamaño de las etiquetas en los ejes x e y
plt.xticks(fontsize=8)
plt.grid(True, alpha=0.2) # Agregar una cuadrícula al gráfico con transparencia
plt.show() # Mostrar el gráfico

# En base a lo obtenido en el metodo del codo se elige probar varios valores de k
modeloB_k1 = 10

# Generar modelo para el AnexoA con cierto valor de k
kmeansB1 = KMeans (n_clusters= modeloB_k1).fit(datos_normB.values)
labels = kmeansB1.labels_
labels = kmeansB1.predict(datos_normB.values)

datos_normB1 = datos_normB
datos_normB1["cluster"]= kmeansB1.labels_ # Nueva columna con los grupos
B_k1 = pd.concat([datosB_i, datos_normB1], axis=1)  # Uniendo colunmas
#print(B_k1) # Imprimiendo tabla completa

print("Imprimiento los centroides para k=", modeloB_k1 )
print("Centroides:\n", kmeansB1.cluster_centers_)
print("Inercia:", kmeansB1.inertia_)

# Plot the data points and their cluster assignments
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(datos_normB.values[:, 0], datos_normB.values[:, 1], datos_normB.values[:, 2], c=labels, cmap='viridis')
ax.scatter(kmeansB1.cluster_centers_[:, 0], kmeansB1.cluster_centers_[:, 1], kmeansB1.cluster_centers_[:, 2], marker='x', color='red', s=100 , linewidths=3)
ax.set_title("K-means Clustering")
ax.set_xlabel('Abonados')
ax.set_ylabel('DPI')
ax.set_zlabel('FPI')
plt.show()

