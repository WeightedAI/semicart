import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeightCalculator:
    def __init__(self, n_neighbors=3, algorithm='auto', n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_jobs = n_jobs

    def calculate_weights_nn(self, x_train, x_test, weight=1):
        near_neighbor = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, n_jobs=self.n_jobs).fit(x_train)
        distances, indices = near_neighbor.kneighbors(x_test)
        training_record_weights = np.ones(len(x_train))

        rows = indices.flatten()
        data = np.full_like(rows, weight, dtype=np.float32)
        weight_updates = csr_matrix((data, (rows, np.zeros_like(rows))), shape=(len(x_train), 1))

        training_record_weights += np.squeeze(np.array(weight_updates.sum(axis=1)))
        return training_record_weights

    def _calculate_distances(self, x_train, x_test, measure_type):
        return cdist(x_train, x_test, metric=measure_type)

    def calculate_weights_dist(self, x_train, x_test, weight=1, measure_type='euclidean'):
        distances = self._calculate_distances(x_train, x_test, measure_type)
        training_record_weights = np.ones(len(x_train))

        nearest_neighbors_indices = np.argpartition(distances, self.n_neighbors, axis=0)[:self.n_neighbors, :]

        if not isinstance(weight, (float, int)):
            logging.error(f"Weight '{weight}' is not numeric. Setting weight to default 0.1.")
            weight = 0.1

        def update_weights(i):
            nearest_indices = nearest_neighbors_indices[:, i]
            return nearest_indices

        updated_indices = Parallel(n_jobs=self.n_jobs)(delayed(update_weights)(i) for i in range(len(x_test)))

        for nearest_indices in updated_indices:
            training_record_weights[nearest_indices] += weight

        return training_record_weights

    def get_measurements(self):
        return [
            'euclidean', 'hamming', 'jaccard', 'braycurtis', 'canberra', 'chebyshev',
            'cityblock', 'correlation', 'cosine', 'dice', 'jensenshannon', 'kulczynski1',
            'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
            'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        ]
