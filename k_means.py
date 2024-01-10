import numpy as np
import matplotlib.pyplot as plt



def k_mean(samples, num_clusters, stop_epsion=1e-2, max_iter=100):
    sample_cluster_index==np.zeros(samples.shape[0], dtype.np.int)
    print("sample shale", sample.shape)
    print("sample cluster index", sample_cluster_index)

    sample_cluster_distances=np.zeros((num_clusters, sample.shape[0]), dtype.np.float32)
    print("sample cluster distances", sample_cluster_distances)

    random_indices=npm.arrange(sample.shape[0],dtype.np.int)
    np.random.shuffle(random_indices)
    cluster_loc=sample[random_indices[: num_clusters],:]
    old_distance_var=-10000;

