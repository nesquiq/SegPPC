"""Mean shift clustering algorithm.

Mean shift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.
"""

# Author Mengyang Zhao <Mengyang.Zhao@tufts.edu>

# Based on: Conrad Lee <conradlee@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@inria.fr>
#           Gael Varoquaux <gael.varoquaux@normalesup.org>
#           Martino Sorbaro <martino.sorbaro@ed.ac.uk>

import numpy as np
import warnings
import math
import os
import torch
import torch.nn.functional as F

from collections import defaultdict
#from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, gen_batches, check_array
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances_argmin
from joblib import Parallel
from joblib import delayed

# from meanshift.batch_seed import meanshift_torch
from FMSEuc.meanshift.batch_seed import meanshift_torch
# from batch_seed import meanshift_torch
from random import shuffle

#seeds number intital
SEED_NUM = 128
L=8
H=32

def KNN(X, target, k):
    dist_mat = torch.cdist(X, target)
    dist, idx = dist_mat.topk(k=k, dim=0, largest=False)
    return dist, idx

def KNN_cosine(X, target, k):
    X = F.normalize(X,-1)
    target = F.normalize(target,-1)
    sim_mat = torch.mm(X, target.T)
    dist, idx = sim_mat.topk(k=k, dim=0, largest=False)
    return dist, idx

# def rad_neighbor_cosine(X, target, k ,rad):
#     dist_mat = torch.cdist(X, target.unsqueeze(0).cuda())
#     dist, idx = dist_mat.topk(k=k, dim=0, largest=False)
#     idx_in_rad = idx[torch.where(dist<rad)]
#     return dist, idx_in_rad

def rad_neighbor_cosine(X, target, k ,rad):
    X = F.normalize(X,dim=-1)
    target = F.normalize(target,dim=-1)
    sim_mat = torch.mm(X, target.T)
    dist_mat = -1*(sim_mat-1)
    # dist, idx = dist_mat.topk(k=k, dim=0, largest=False)
    # idx_in_rad = idx[torch.where(dist<rad)]
    idx_in_rad = torch.where(dist_mat<rad)[0]
    return dist_mat, idx_in_rad

def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    """Estimate the bandwidth to use with the mean-shift algorithm.

    That this function takes time at least quadratic in n_samples. For large
    datasets, it's wise to set that parameter to a small value.

    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input points.

    quantile : float, default 0.3
        should be between [0, 1]
        0.5 means that the median of all pairwise distances is used.

    n_samples : int, optional
        The number of samples to use. If not given, all samples are used.

    random_state : int, RandomState instance or None (default)
        The generator used to randomly select the samples from input points
        for bandwidth estimation. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    bandwidth : float
        The bandwidth parameter.
    """
    ## is np ndarray? is nan in array?
    X = check_array(X)

    ## check random state for n_sample < total_sample
    random_state = check_random_state(random_state)
    if n_samples is not None:
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]

    ## n_neighbors
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:  
        n_neighbors = 1

    ## nn define
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            n_jobs=n_jobs)
    nbrs.fit(X)

    ## initial, bw
    bandwidth = 0.

    ## slice total sample with size 500
    ## sum bw 3 times
    for batch in gen_batches(len(X), 500):

        ## (batch <---> total_sample) NN with n_neighbors(450)
        ## get distance between batch and total_sample's n_nearest_neighbors
        ## 500, 256 <---> 1500, 256
        ## batch, distance to 450 neighbors --> 500, 450
        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
        ## sum(max distance per row)
        bandwidth += np.max(d, axis=1).sum()

    ## normalize
    return bandwidth / X.shape[0]


def gpu_seed_generator(codes):

    ## random seed indices
    seed_indizes = list(range(codes.shape[0]))
    shuffle(seed_indizes)
    
    seed_indizes = seed_indizes[:SEED_NUM]
    seeds = codes[seed_indizes]
    
    return seeds
    
def gpu_seed_generator_gpu(codes):

    ## random seed indices
    seed_indizes = list(range(codes.shape[0]))
    shuffle(seed_indizes)
    
    # ## fixed seed indices
    # seed_indizes = np.arange(SEED_NUM+1)[1:]
    # seed_indizes = seed_indizes*(codes.shape[0]//SEED_NUM) -1

    seed_indizes = seed_indizes[:SEED_NUM]
    seeds = codes[seed_indizes]
    
    return seeds

def gpu_seed_adjust(codes):
    global SEED_NUM
    SEED_NUM *= 2
    
    # return gpu_seed_generator(codes)
    return gpu_seed_generator_gpu(codes)

def get_N(P,r,I):

    #There is no foreground instances
    if r<0.1:
        return 32 #Allocated some seeds at least

    lnp = math.log(P,math.e)
    num=math.log(1-math.e**(lnp/I),math.e)
    den = math.log(1-r/I,math.e)
    result = num/den

    if result<32:
        result =32 #Allocated some seeds at least
    elif result>256:
        result =256 #Our GPU memory's max limitation, you can higher it.

    return int(result)


def mean_shift_euc(X, X_gpu, bandwidth=None, seeds=None, 
                      cluster_all=True, GPU=True):
    """Perform mean shift clustering of data using a flat kernel.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------

    X : array-like, shape=[n_samples, n_features]
        Input data.

    bandwidth : float, optional
        Kernel bandwidth.

        If bandwidth is not given, it is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.

    seeds : array-like, shape=[n_seeds, n_features] or None
        Point used as initial kernel locations. 

    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    GPU : bool, default True
        Using GPU-based faster mean-shift


    Returns
    -------

    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.

    labels : array, shape=[n_samples]
        Cluster labels for each point.


    """

    if bandwidth is None:
        bandwidth = estimate_bandwidth(X)
    elif bandwidth <= 0:
        raise ValueError("bandwidth needs to be greater than zero or None,\
            got %f" % bandwidth)
    if seeds is None:
        if GPU == True:
            # seeds2 = gpu_seed_generator(X)
            seeds  = gpu_seed_generator_gpu(X_gpu)
            
    
    #adjusted=False
    n_samples, n_features = X.shape
    center_intensity_dict = {}
    labels_torch = torch.tensor([])
    labels_torch = labels_torch.reshape(0, 256)
    number_torch = torch.tensor([])
    number_torch = number_torch.reshape(0, 1)

    global SEED_NUM
    if GPU == True:
        #GPU ver
        while True:
            # labels, number = meanshift_torch(X, X_gpu, seeds, seeds2, bandwidth)#gpu calculation
            
            labels, number = meanshift_torch(None, X_gpu, seeds, bandwidth)#gpu calculation
            # labels, number = meanshift_torch(X, X_gpu, seeds, bandwidth)#gpu calculation
            for i in range(len(number)):
                if number[i] is not None:
                    center_intensity_dict[tuple(labels[i])] = number[i]#find out cluster

            if not center_intensity_dict:
                # nothing near seeds
                raise ValueError("No point was within bandwidth=%f of any seed."
                            " Try a different seeding strategy \
                             or increase the bandwidth."
                            % bandwidth)
                            
            # POST PROCESSING: remove near duplicate points
            # If the distance between two kernels is less than the bandwidth,
            # then we have to remove one because it is a duplicate. Remove the
            # one with fewer points.
            
            ## 일정 bw 이내의 center들 합치기
            ## stable sort with (intensity, label)
            sorted_by_intensity = sorted(center_intensity_dict.items(),
                                        key=lambda tup: (tup[1], tup[0]),
                                        reverse=True)
            sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
            """
            labels_torch = torch.tensor(labels)
            number_torch = torch.tensor(number)
            tor = torch.concat([number_torch.reshape(-1,1),labels_torch],-1)
            _, id1 = torch.sort(tor[:,0], descending=True)
            tor_sort = tor[id1]
            _, id2 = torch.sort(tor_sort[:,1], descending=True, stable=True)
            tor_sort = tor_sort[id2]
            sorted_cs = tor_sort[:,1:]

            unique_torch = torch.ones(sorted_cs.shape[0], dtype=torch.bool)
            for i, c in enumerate(sorted_cs):
                if unique_torch[i]:
                    _, neighbor_idxs = rad_neighbor_cosine(sorted_centers,centers.unsqueeze(0),5,bandwidth)
                    unique_torch[neighbor_idxs] = 0
                    unique_torch[i] = 1
            cluster_centers = sorted_cs[unique_torch]
            """

            labels_torch = torch.cat([labels_torch,torch.tensor(labels)],0)
            number_torch = torch.cat([number_torch,torch.tensor(number).reshape(-1,1)],0)
            tor = torch.concat([number_torch,labels_torch],-1)
            _, id1 = torch.sort(tor[:,0], descending=True)
            tor_sort = tor[id1]
            _, id2 = torch.sort(tor_sort[:,1], descending=True, stable=True)
            tor_sort = tor_sort[id2]
            sorted_cs = tor_sort[:,1:]

            ## 마스크로 사용예정
            unique = np.ones(len(sorted_centers), dtype=np.bool)
            unique_torch = torch.ones(sorted_cs.shape[0], dtype=torch.bool)
            ## nn with bandwidth, n_neighbors 5, metric=cosine_similarity
            nbrs = NearestNeighbors(radius=bandwidth, metric='cosine').fit(sorted_centers)
            # sorted_centers = torch.from_numpy(sorted_centers)
            ## 정렬된 센터 순서대로 집어넣으면서
            for i, center in enumerate(sorted_centers):
                if unique[i]:
                    # _, neighbor_idxs = rad_neighbor(sorted_centers,center,5,bandwidth)
                    
                    ## sorted_center에서 radius 이내의 점들의 idx 가져옴
                    # distance, neighbor_idxs = nbrs.radius_neighbors([center],
                    #                                 return_distance=True)
                    neighbor_idxs = nbrs.radius_neighbors([center],
                                                    return_distance=False)[0]
                    ## 해당 idx의 마스크 0
                    unique[neighbor_idxs] = 0
                    dd, nn = rad_neighbor_cosine(sorted_cs,sorted_cs[i].unsqueeze(0),5,bandwidth)
                    unique_torch[nn] = 0
                    ## 현재 포인트 마스크 1
                    unique[i] = 1  # leave the current point as unique
                    unique_torch[i] = 1
            cluster_centers = sorted_centers[unique]
            clst_centers = sorted_cs[unique_torch]
            print('unique_sanity',torch.equal(unique_torch,torch.from_numpy(unique)))

            ## 남은 센터들에 1 nn으로 모든 점들 레이블링
            # assign labels
            nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
            out_labels = np.zeros(n_samples, dtype=np.int)
            distance, idx = KNN(clst_centers.cuda(), X_gpu, 1)
            distances, idxs = nbrs.kneighbors(X)
            # bw 밖의 orphan을 버릴 것인가 아니면 그래도 nn으로 가져갈 것인가
            if cluster_all:
                out_labels_torch = idx.flatten()
                out_labels = idxs.flatten()
            else:
                out_labels.fill(-1)
                bool_selector = distances.flatten() <= bandwidth
                out_labels[bool_selector] = idxs.flatten()[bool_selector]
            
            print('label_sanity', torch.equal(out_labels_torch,torch.from_numpy(out_labels).cuda()))
            bg_num = np.sum(out_labels==0)
            r = 1-bg_num/out_labels.size
            #seed number adjust
            dict_len = len(cluster_centers)#cluster number

            M = dict_len

            # SAFE_UP = SEED_NUM/4
            # SAFE_UP = 12
            SAFE_UP = 64
            # SAFE_DOWN = SEED_NUM/8
            # SAFE_DOWN = 4
            SAFE_DOWN = 32

            # if SAFE_DOWN <= M <=SAFE_UP: #safety area
            #     #SEED_NUM -= 200#test
            #     #if H*M  <= SEED_NUM:
            #     #    SEED_NUM -= M #seeds are too much, adjsut
            #     # print('brk', M)
            #     break
            # elif M < SAFE_DOWN:
            #     # seeds = gpu_seed_adjust(X_gpu)#seeds are too few, adjsut
            #     # print('adj_d', M)
            #     bandwidth *= 0.8 # bw is too big, adjsut --> v02
            #     # break
            # else: # SAFE_UP < M
            #     # seeds = gpu_seed_adjust(X_gpu)#seeds are too few, adjsut
            #     # print('adj_u', M)
            #     bandwidth *= 1.2 # bw is too small, adjsut --> v02
            #     # break
            # break
            if M <=SAFE_UP and SAFE_DOWN <= M: #safety area
            # if SAFE_DOWN <= M <=SAFE_UP: #safety area double ended
                #SEED_NUM -= 200#test
                #if H*M  <= SEED_NUM:
                #    SEED_NUM -= M #seeds are too much, adjsut
                # print('brk', M)
                break
            elif M < SAFE_DOWN:
                # seeds = gpu_seed_adjust(X_gpu)#seeds are too few, adjsut
                # print('adj_d', M)
                bandwidth *= 0.7 # bw is too big, adjsut --> v02
                # break
            else: # SAFE_UP < M
                # seeds = gpu_seed_adjust(X_gpu)#seeds are too few, adjsut
                # print('adj_u', M)
                bandwidth *= 1.5 # bw is too small, adjsut --> v02
                # break
        
        return cluster_centers, out_labels



class MeanShiftEuc(BaseEstimator, ClusterMixin):
    """Mean shift clustering using a flat kernel.

    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.

    Seeding is performed using a binning technique for scalability.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------
    bandwidth : float, optional
        Bandwidth used in the RBF kernel.

        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).

    seeds : array, shape=[n_samples, n_features], optional
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.

    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    GPU : bool, default True
        Using GPU-based faster mean-shift


    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ :
        Labels of each point.

    Examples
    --------
    >>> from sklearn.cluster import MeanShift
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = MeanShift(bandwidth=2).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering.predict([[0, 0], [5, 5]])
    array([1, 0])
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    MeanShift(bandwidth=2, cluster_all=True, seeds=None)

    References
    ----------

    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.

    """
    def __init__(self, bandwidth=None, seeds=None, cluster_all=True, GPU=True):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.cluster_all = cluster_all
        self.GPU = GPU

    def fit(self, X, X_gpu, y=None):
        """Perform clustering.

        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.

        y : Ignored

        """
        X = check_array(X)
        self.cluster_centers_, self.labels_ = \
            mean_shift_euc(X, X_gpu, bandwidth=self.bandwidth, seeds=self.seeds,
                       cluster_all=self.cluster_all, GPU=self.GPU)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")

        return pairwise_distances_argmin(X, self.cluster_centers_)
