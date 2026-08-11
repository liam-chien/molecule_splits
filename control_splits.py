import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import random
from pathlib import Path

import umap.umap_ as umap
from sklearn.cluster import KMeans, SpectralClustering


def random_split(mols, val_p, seed):

    assert val_p <= 1
    assert val_p >= 0

    total_len = len(mols)
    val_len = round(total_len * val_p)
    
    rng = np.random.default_rng(seed=seed)
    
    val = rng.choice(total_len, val_len, replace=False)

    train = [i for i in range(total_len) if not i in val]

    return train, list(val)



# Scaffold and UMAP split sourced from https://github.com/vynkdo/SPECTRA_evaluation/blob/main/generate_data/create_random_scaffold_umap_splits.py line 58
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
def scaffold_split(mols, val_p, seed, scaffold_to_ix_map, tol=0.01, verbose=False):

    random.seed(seed)

    train_prop = 1 - val_p

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

def umap_split(mfps, val_p, seed, n_clusters=7, save_plot_path=None):
    
    
    test_size = round(val_p * len(mfps))

    if save_plot_path is not None:
        save_plot_path = Path(save_plot_path)
        if save_plot_path.parent != Path(""):
            os.makedirs(save_plot_path.parent, exist_ok=True)

        
    mfp_umap = umap.UMAP(n_neighbors = 15, n_components = 2, transform_seed = seed).fit_transform(mfps)
    kmeans = KMeans(n_clusters = n_clusters, random_state = seed)

    cluster_labels = kmeans.fit_predict(mfp_umap)

    # s_results = []
    # s_score = silhouette_score(mfp_umap, cluster_labels)
    # s_results.append({'dataset': f'{dataset_name}_{i}',
    #                   'silhouette_score': {s_score}})

    # plt.figure(figsize=(6,6))
    # plt.scatter(mfp_umap[:, 0], mfp_umap[:, 1], c = cluster_labels)
    # plt.title(f"{dataset_name} - UMAP embedding {i}")
    # plt.xlabel("UMAP-1")
    # plt.ylabel("UMAP-2")
    # plt.savefig(os.path.join(umap_save_dir_plot, f"{dataset_name}_umap_{i}.png"), dpi=400)
    # plt.close()

    cluster_index, counts = np.unique(cluster_labels, return_counts=True)
    difference_list = [abs(test_size - j) for j in counts]
    
    min_cluster_index = difference_list.index(min(difference_list))
    
    umap_test_indices = (np.where(cluster_labels == min_cluster_index)[0]).tolist()
    umap_train_indices = (np.where(cluster_labels != min_cluster_index)[0]).tolist()

    assert len(set(umap_train_indices) & set(umap_test_indices)) == 0
    assert len(set(np.concatenate([umap_train_indices, umap_test_indices]))) == len(mfps)
        
    # silhouette_df = pd.DataFrame(s_results)
    # s_results_save_dir = os.path.join('splits_data', 'umap_silhouette_results')
    # os.makedirs(s_results_save_dir)
    # silhouette_df.to_csv(os.path.join(s_results_save_dir, f'{dataset_name}_silhouette_scores.csv'))
    
    # print(f"UMAP splits {dataset_name} done.")
    
    # with open(f"splits/{NAME}_umap_0", "r") as file:
    #     umap_split = json.load(file)
        
    # print(f"UMAP_{i}: ", adj_mat_cso(ADJ_MAT, umap_split["train"], umap_split["val"]))


    return umap_train_indices, umap_test_indices


# Important: This version takes BINARY data labels and will break if there are not only 0 or 1 in the data_labels iterable
def spectral_split(adj_mat, data_labels, seed, cluster_count=5):

    assert len(adj_mat) == len(data_labels), "Incorrect data labels for provided affinity matrix"
    
    clustering = SpectralClustering(n_clusters=cluster_count,
            affinity="precomputed",
            random_state=seed).fit(adj_mat)

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


    
    val = []
    cluster_rank = 0
    cluster_order = np.argsort(cluster_scores)
    
    while(len(val) < 0.2 * len(data_labels)):
        # Pick the next-smallest cluster
        next_cluster_i = cluster_order[cluster_rank]
        val.extend(clusters[next_cluster_i])
        cluster_rank += 1

    train = [x for x in range(len(data_labels)) if not x in set(val)]
    
    return train, val
