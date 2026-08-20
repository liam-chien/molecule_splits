import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import random
from pathlib import Path

import umap.umap_ as umap
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.utils.validation import check_random_state


def random_split(mols, test_p, seed):

    assert test_p <= 1
    assert test_p >= 0

    total_len = len(mols)
    test_len = round(total_len * test_p)
    
    rng = np.random.default_rng(seed=seed)
    
    test = rng.choice(total_len, test_len, replace=False)

    train = [i for i in range(total_len) if not i in test]

    return train, list(test)



# Scaffold and UMAP split sourced from https://github.com/vynkdo/SPECTRA_evaluation/blob/main/generate_data/create_random_scaffold_umap_splits.py 


def preprocess_scaffold(mols):
    
    scaffolds = []

    for mol in mols:
        scaffolds.append(Chem.MolToSmiles(GetScaffoldForMol(mol)))

    scaffold_to_ix_map = {
        scaf: [ix for ix, s in enumerate(scaffolds) if s == scaf] for scaf in scaffolds
    }

    return scaffold_to_ix_map
                                                        # tolerance on proportion away 
                                                        # from desired train proportion
def scaffold_split(mols, test_p, seed, scaffold_to_ix_map, tol=0.01, verbose=False):

    random.seed(seed)

    train_prop = 1 - test_p

    train_indices = []
    test_indices = []

    while abs((len(train_indices) / len(mols)) - train_prop) > tol:
        train_indices = []
        test_indices = []
        shuffled_unique_scaffolds = random.sample(list(scaffold_to_ix_map.keys()), k=len(scaffold_to_ix_map))

        while len(train_indices) < (train_prop * len(mols)):
            cur_scaf = shuffled_unique_scaffolds.pop(0)
            train_indices += scaffold_to_ix_map[cur_scaf]

        for remaining_scaffolds in shuffled_unique_scaffolds:
            test_indices += scaffold_to_ix_map[remaining_scaffolds]

        if verbose:
            print(f"Train proportion: {len(train_indices) / len(mols)}")
            print(f"Test proportion: {len(test_indices) / len(mols)}")

    assert len(set(train_indices) & set(test_indices)) == 0
    assert len(set(train_indices + test_indices)) == len(mols)

    return train_indices, test_indices




# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt

def umap_split(mfps, test_p, seed, n_clusters=7, save_plot_path=None):
    
    
    test_size = round(test_p * len(mfps))

    if save_plot_path is not None:
        save_plot_path = Path(save_plot_path)
        if save_plot_path.parent != Path(""):
            os.makedirs(save_plot_path.parent, exist_ok=True)

        
    mfp_umap = umap.UMAP(n_neighbors = 15, n_components = 2, transform_seed = seed).fit_transform(mfps)
    kmeans = KMeans(n_clusters = n_clusters, random_state = seed)

    cluster_labels = kmeans.fit_predict(mfp_umap)

    # Can be used to save the embeddings
    # embeddings = (mfp_umap[:, 0], mfp_umap[:, 1])

    cluster_index, counts = np.unique(cluster_labels, return_counts=True)
    difference_list = [abs(test_size - j) for j in counts]
    
    min_cluster_index = difference_list.index(min(difference_list))
    
    umap_test_indices = (np.where(cluster_labels == min_cluster_index)[0]).tolist()
    umap_train_indices = (np.where(cluster_labels != min_cluster_index)[0]).tolist()

    assert len(set(umap_train_indices) & set(umap_test_indices)) == 0
    assert len(set(np.concatenate([umap_train_indices, umap_test_indices]))) == len(mfps)
        

    return umap_train_indices, umap_test_indices


# Important: This version takes BINARY data labels and will break if there are not only 0 or 1 in the data_labels iterable
# Code referenced from https://github.com/arunraja-hub/quadmetformer/blob/b5789c52a2701009005b98930fae24f8c136142f/finetuning/preprocessing_tdc_dataset.py line 1159
def spectral_split(adj_mat, data_labels, test_p, seed):

    random_state = check_random_state(seed)

    cluster_count = int(1 / test_p)


    assert len(adj_mat) == len(data_labels), "Incorrect data labels for provided affinity matrix"
    
    clustering = SpectralClustering(
        n_clusters=cluster_count,
        assign_labels='kmeans',
        n_init=10,
        random_state=random_state,
        affinity="precomputed").fit(adj_mat)

    cluster_assignments = clustering.labels_
    clusters = []
    
    for x in range(cluster_count):
        clusters.append(set(np.asarray(cluster_assignments == x).nonzero()[0]))
    
    # This will be populated by the lb scores of the clusters
    cluster_scores = np.zeros(cluster_count)

    data_dist = np.count_nonzero(data_labels == 1) / len(data_labels)

    for x, c in enumerate(clusters):

        cluster_dist = np.count_nonzero(data_labels[list(c)] == 1) / len(c)

        # Label-balance score
        lb_score = abs(cluster_dist - data_dist)
        
        cluster_scores[x] = lb_score

    cluster_order = np.argsort(cluster_scores)

    # Begin the test set with the best cluster
    best_cluster_idx = cluster_order[0]
    test = list(clusters[best_cluster_idx])

    # Add next-best clusters...
    cluster_rank = 1
    
    while(True):
        next_cluster_i = cluster_order[cluster_rank]
        next_cluster = clusters[next_cluster_i]

        # ... until the target size would be exceeded
        if (len(test) + len(next_cluster) > test_p * len(data_labels)):
            break

        test.extend(next_cluster)
        cluster_rank += 1

        # Could also stop here... increases CSO for the BACE dataset at least
        # if (len(test) > test_p * len(data_labels)):
        #     break

    train = [x for x in range(len(data_labels)) if x not in test]
    
    return train, test
