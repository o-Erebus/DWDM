import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

data = pd.read_csv("Iris.csv")
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

Z = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(Z, labels=data['Species'].values, leaf_rotation=90, leaf_font_size=8)
plt.title("Dendrogram")
plt.xlabel("Species")
plt.ylabel("Distance")
plt.show()
