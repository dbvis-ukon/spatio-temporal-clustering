from src.st_clustering.st_clustering import *

df = np.array([
        [1, 0],
        [1, 0],
        [12, 4],
        [12, 10],
        [12, 10]
    ])

clusterer = ST_KMeans(n_clusters=2)
a = clusterer.st_fit(df)
a.labels
