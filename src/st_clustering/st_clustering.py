import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, OPTICS, SpectralClustering, AffinityPropagation, Birch
from sklearn.utils import check_array
import hdbscan
from joblib import Memory
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.neighbors import NearestNeighbors
import warnings


def st_decorator(target):

    def st_fit(self, X):
        """
        Apply the ST clustering algorithm 
        ----------
        X : 2D numpy array with
            The first element of the array should be the time 
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        Returns
        -------
        self
        """
        # call sparse matrix method for larger inputs and DBSCAN
        if len(X) > 10000 and type(self).__name__ == 'ST_DBSCAN':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st_fit_sparsematrix(self, X)
                return self

        # check if input is correct
        X = check_array(X)

        if not self.eps1 > 0.0 or not self.eps2 > 0.0:
            raise ValueError('eps1, eps2 must be positive')

        n, m = X.shape

        # Compute sqaured form Euclidean Distance Matrix for 'time' attribute and the spatial attributes
        time_dist = pdist(X[:, 0].reshape(n, 1), metric=self.dist)
        euc_dist = pdist(X[:, 1:], metric=self.dist)

        # filter the euc_dist matrix using the time_dist
        dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)

        self.fit(squareform(dist))

        self.labels = self.labels_

        return self

    def st_fit_sparsematrix(self, X):
        """
        Fit method for larger input arrays: instead of a dense distance matrix for performance reasons only a sparse
        distance matrix is calculated and used for the DBSCAN clustering algorithm.
        """
        # check if input is correct
        X = check_array(X)

        if not self.eps1 > 0.0 or not self.eps2 > 0.0:
            raise ValueError('eps1, eps2 must be positive')

        n, m = X.shape

        # create sparse distance matrix for time attribute
        neigh2 = NearestNeighbors(metric='euclidean', radius=self.eps2)
        neigh2.fit(X[:, 0].reshape(n, 1))
        B = neigh2.radius_neighbors_graph(X[:, 0].reshape(n, 1),
                                          mode='distance')

        # create sparse distance matrix for spatial attributes
        neigh = NearestNeighbors(metric='euclidean', radius=self.eps1)
        neigh.fit(X[:, 1:])
        A = neigh.radius_neighbors_graph(X[:, 1:], mode='distance')

        # store values to create new sparse distance matrix for spatial attributes filtered due to time distance matrix
        row = B.nonzero()[0]
        column = B.nonzero()[1]
        v = np.array(A[row, column])[0]

        # create ney sparse distance matrix for spatial attributes
        precomputed_matrix = coo_matrix(
            (v, (row, column)),
            shape=(n, n))  # sparse matrix format more efficient for creation
        precomputed_matrix = precomputed_matrix.tocsc(
        )  # convert to sparse matrix format more efficient for matrix computations
        precomputed_matrix.eliminate_zeros(
        )  # to delete matrix entries which were non-zero in time matrix but zero in spatial matrix

        self.fit(precomputed_matrix)

        self.labels = self.labels_

        return self

    def st_fit_frame_split(self, X, frame_size, frame_overlap=None):
        """
        Apply the ST clustering algorithm with splitting it into frames.
        ----------
        X : 2D numpy array with
            The first element of the array should be the time (sorted by time)
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        frame_size : float, default= None
            If not none the dataset is split into frames and merged aferwards
        frame_overlap : float, default=eps2
            If frame_size is set - there will be an overlap between the frames
            to merge the clusters afterwards 
        Returns
        -------
        self
        """

        # check if input is correct
        X = check_array(X)

        # default values for overlap
        if frame_overlap == None:
            frame_overlap = self.eps2  # in the paper they use 2*self.eps2

        if not self.eps1 > 0.0 or not self.eps2 > 0.0:
            raise ValueError('eps1, eps2 must be positive')

        if not frame_size > 0.0 or not frame_overlap > 0.0 or frame_size < frame_overlap:
            raise ValueError(
                'frame_size, frame_overlap not correctly configured.')

        # unique time points
        time = np.unique(X[:, 0])

        labels = None
        right_overlap = 0
        max_label = 0

        # iterate over frames
        for i in range(0, len(time), (frame_size - frame_overlap + 1)):
            for period in [time[i:i + frame_size]]:
                frame = X[np.isin(X[:, 0], period)]

                self.st_fit(frame)

                # match the labels in the overlaped zone
                # objects in the second frame are relabeled to match the cluster id from the first frame
                if not type(labels) is np.ndarray:
                    labels = self.labels
                else:
                    frame_one_overlap_labels = labels[len(labels) -
                                                      right_overlap:]
                    frame_two_overlap_labels = self.labels[0:right_overlap]

                    mapper = {}
                    for j in list(
                            zip(frame_one_overlap_labels,
                                frame_two_overlap_labels)):
                        mapper[j[1]] = j[0]
                    mapper[-1] = -1  # to  avoid outliers being mapped on a cluster

                    # clusters without points in the overlapping area are given new cluster
                    # otherwise, there will be a key error
                    ignore_clusters = set(self.labels) - set(
                        frame_two_overlap_labels)  # set difference
                    # recode them to new cluster value
                    if -1 in labels:
                        labels_counter = len(set(labels)) - 1
                    else:
                        labels_counter = len(set(labels))
                    for j in ignore_clusters:
                        mapper[j] = labels_counter
                        labels_counter += 1

                    # objects in the second frame are relabeled to match the cluster id from the first frame
                    # objects in clusters with no overlap are assigned to new clusters
                    new_labels = np.array([
                        mapper[j] for j in self.labels
                    ])

                    # delete the right overlap
                    labels = labels[0:len(labels) - right_overlap]
                    # change the labels of the new clustering and concat
                    labels = np.concatenate((labels, new_labels))

                right_overlap = len(X[np.isin(X[:, 0],
                                              period[-frame_overlap + 1:])])

            if i + frame_size > max(
                    time
            ):  # we need that condition, otherwise we'll do an unnessecary iteration
                break

        self.labels = labels
        return self

    target.st_fit = st_fit
    target.st_fit_frame_split = st_fit_frame_split
    target.st_fit_sparsematrix = st_fit_sparsematrix
    return target


@st_decorator
class ST_DBSCAN(DBSCAN):
    """
    A class to perform the ST_DBSCAN clustering
    Parameters
    ----------
    eps1 : float, default=0.5
        The spatial density threshold (maximum spatial distance) between 
        two points to be considered related.
    eps2 : float, default=10
        The temporal threshold (maximum temporal distance) between two 
        points to be considered related.
    min_samples : int, default=5
        The number of samples required for a core point.
    dist : string default='euclidean'
        The used distance metric - more options are
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
    n_jobs : int or None, default=-1
        The number of processes to start -1 means use all processors 
    Attributes
    ----------
    labels : array, shape = [n_samples]
        Cluster labels for the data - noise is defined as -1
    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Birant, Derya, and Alp Kut. "ST-DBSCAN: An algorithm for clustering spatial–temporal data." Data & Knowledge Engineering 60.1 (2007): 208-221.
    
    Peca, I., Fuchs, G., Vrotsou, K., Andrienko, N. V., & Andrienko, G. L. (2012). Scalable Cluster Analysis of Spatial Events. In EuroVA@ EuroVis.
    """

    # overwrite sklearn's constructor
    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 min_samples=5,
                 metric='precomputed',
                 n_jobs=-1,
                 algorithm='auto',
                 leaf_size=30,
                 metric_params=None,
                 p=None,
                 dist='euclidean'):
        self.eps = eps1
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric_params = metric_params
        self.p = p
        self.dist = dist


@st_decorator
class ST_Agglomerative(AgglomerativeClustering):

    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 n_clusters=2,
                 *,
                 affinity='precomputed',
                 memory=None,
                 connectivity=None,
                 compute_full_tree='auto',
                 linkage='average',
                 distance_threshold=None,
                 compute_distances=False,
                 dist='euclidean'):
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.affinity = affinity
        self.compute_distances = compute_distances
        self.dist = dist


"""
commented out as kmeans takes feature array and no distance matrix as input
@st_decorator
class ST_KMeans(KMeans):
    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 n_clusters=8,
                 *,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 precompute_distances='deprecated',
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 n_jobs='deprecated',
                 algorithm='auto',
                 dist='euclidean'):
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.dist = dist
"""


@st_decorator
class ST_OPTICS(OPTICS):

    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 min_samples=5,
                 max_eps=np.inf,
                 metric='precomputed',
                 p=2,
                 metric_params=None,
                 cluster_method='xi',
                 eps=None,
                 xi=0.05,
                 predecessor_correction=True,
                 min_cluster_size=None,
                 algorithm='auto',
                 leaf_size=30,
                 n_jobs=-1,
                 dist='euclidean'):
        self.eps1 = eps1
        self.eps2 = eps2
        self.max_eps = max_eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.n_jobs = n_jobs
        self.dist = dist


@st_decorator
class ST_SpectralClustering(SpectralClustering):

    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 n_clusters=8,
                 *,
                 eigen_solver=None,
                 n_components=None,
                 random_state=None,
                 n_init=10,
                 gamma=1.,
                 affinity='precomputed',
                 n_neighbors=10,
                 eigen_tol=0.0,
                 assign_labels='kmeans',
                 degree=3,
                 coef0=1,
                 kernel_params=None,
                 n_jobs=None,
                 verbose=False,
                 dist='euclidean'):
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.dist = dist


@st_decorator
class ST_AffinityPropagation(AffinityPropagation):

    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 damping=.5,
                 max_iter=200,
                 convergence_iter=15,
                 copy=True,
                 preference=None,
                 affinity='precomputed',
                 verbose=False,
                 random_state='warn',
                 dist='euclidean'):
        self.eps1 = eps1
        self.eps2 = eps2
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state
        self.dist = dist


"""
commented out as Birch takes feature array and no distance matrix as input
@st_decorator
class ST_BIRCH(Birch):
    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 threshold=0.5,
                 branching_factor=50,
                 n_clusters=3,
                 compute_labels=True,
                 copy=True,
                 dist='euclidean'):
        self.eps1 = eps1
        self.eps2 = eps2
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy
        self.dist = dist
"""


@st_decorator
class ST_HDBSCAN(hdbscan.HDBSCAN):

    def __init__(
            self,
            eps1=0.5,
            eps2=10,
            min_cluster_size=5,
            min_samples=None,
            cluster_selection_epsilon=0.0,
            max_cluster_size=0,
            metric='precomputed',
            alpha=1.0,
            p=None,  #euclidean 
            algorithm='best',
            leaf_size=40,
            memory=Memory(cachedir=None, verbose=0),
            approx_min_span_tree=True,
            gen_min_span_tree=False,
            core_dist_n_jobs=4,
            cluster_selection_method='eom',
            allow_single_cluster=False,
            prediction_data=False,
            match_reference_implementation=False,
            dist='euclidean',
            **kwargs):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.prediction_data = prediction_data
        self.dist = dist
        self._metric_kwargs = kwargs
