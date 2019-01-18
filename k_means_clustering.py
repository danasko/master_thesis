from sklearn.cluster import KMeans
import numpy as np


def cluster_prototypes(train_labels, K):  # input shape (#samples, J, 3) - train_data labels, K - no. of clusters
    J = train_labels.shape[1]
    train_labels = np.reshape(train_labels, (train_labels.shape[0], J * 3))
    kmeans = KMeans(n_clusters=K, random_state=2)
    kmeans.fit(train_labels)
    kmeans.predict(train_labels)
    centers = kmeans.cluster_centers_

    centers = np.reshape(centers, (K, J, 3))
    centers = np.transpose(centers, (1, 0, 2))
    return centers  # output shape (J, K, 3)
