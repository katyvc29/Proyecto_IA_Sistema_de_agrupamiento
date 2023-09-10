import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # Normalizar
from sklearn.cluster import KMeans


# Leemos el archivo csv del anexoA
data = pd.read_csv("anexoA.csv")
#df2 = pd.read_csv("anexoB.csv")

# print(data)

datos = data[['Abonados', 'DPI', 'FPI']]

# print(dataframe)
# print(data.describe())
#  Calcula los valores mínimos y máximos 
escalador = MinMaxScaler().fit(datos.values)

# Datos normalizados 
# Se aplica la transformación de escala a los datos utilizando el escalador previamente ajustado
datos_norm = pd.DataFrame(escalador.transform(datos.values), columns = ['Abonados', 'DPI', 'FPI'])

#print(datos_norm)

# Generar modelo
kmeans = KMeans (n_clusters=120).fit(datos_norm.values)
kmeans.labels_

#print(kmeans.labels_)

# Nueva columna
datos_norm["cluster"]= kmeans.labels_
#print(datos_norm)

# Imprimir donde estan los centroides, y la inercia que tambien conformados estan los clusters
#print(kmeans.cluster_centers_, kmeans.inertia_)

# Graficacion
inercias = []
for k in range(2, 40):
    kmeans = KMeans(n_clusters=k).fit(datos_norm.values)    
    inercias.append(kmeans.inertia_)

print(inercias)

plt.figure(figsize=(6, 5), dpi=120)
plt.scatter(range(2, 40), inercias, marker="o", s=10, color="blue")
plt.xlabel("Número de Clusters", fontsize=10)
plt.ylabel("Inercia", fontsize=10)
plt.xticks(range(2, 41, 2))
# yticks = list(range(104, 519, 100)) + list(range(593, 2971, 300))+ list(range(5000, 37000, 5000))
#plt.yticks(yticks)
#plt.yticks(range(100, 10000, 500))

# yticks = inercias[0:5] + inercias[9:10]+ [inercias[15]] + inercias[19:20] 
yticks = inercias[0:7] + inercias[9:10] + inercias[19:20] 
plt.yticks(yticks)
plt.yticks(fontsize=5.5) 
plt.xticks(fontsize=8) 
plt.grid(True, alpha=0.2)
plt.show()

