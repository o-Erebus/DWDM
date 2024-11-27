import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("kmeans/Iris.csv")
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

k = 3

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

data['Cluster'] = kmeans.labels_

plt.scatter(X['SepalLengthCm'], X['SepalWidthCm'], c=data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', label = 'Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()