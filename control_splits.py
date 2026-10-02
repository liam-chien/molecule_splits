import random

import numpy as np

import umap.umap_ as umap

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.DataStructs import ExplicitBitVect

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.utils.validation import check_random_state


def random_split(
    mols: list[Mol],
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Generate a random train/test split from the given molecules.

    Only the length of ``mols`` matters for a random split; the molecules
    themselves are not inspected.

    Args:
        mols (list[Mol]): List of RDKit mol objects.
        test_frac (float): The proportion of the dataset that the test set should
            represent. Strictly between 0 and 1.
        seed (int): The random seed.

    Returns:
        tuple[list[int], list[int]]: A tuple containing two mutually exclusive
            lists: (train_indices, test_indices).

    Raises:
        ValueError: If ``test_frac`` is not strictly between 0 and 1.
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    n_mols = len(mols)
    n_test = round(n_mols * test_frac)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_mols)

    return perm[n_test:].tolist(), perm[:n_test].tolist()


# Scaffold and UMAP split sourced from https://github.com/vynkdo/SPECTRA_evaluation/blob/main/generate_data/create_random_scaffold_umap_splits.py

def preprocess_scaffold(mols: list[Mol]) -> dict[str, list[int]]:
    """Group molecule indices by Bemis-Murcko scaffold.

    Args:
        mols (list[Mol]): List of RDKit mol objects.

    Returns:
        dict[str, list[int]]: Dictionary mapping each scaffold SMILES to the
            indices in ``mols`` of the molecules with that scaffold.
    """
    scaffolds = []

    # Get the scaffold SMILES of each molecule
    for mol in mols:
        scaffolds.append(Chem.MolToSmiles(GetScaffoldForMol(mol)))

    # Put (scaffold, [idx0, idx1, ...]) for each scaffold into the dictionary
    scaffold_to_indices = {
        scaf: [ix for ix, s in enumerate(scaffolds) if s == scaf] for scaf in scaffolds
    }

    return scaffold_to_indices


def scaffold_split(
    mols: list[Mol],
    test_frac: float,
    seed: int,
    scaffold_to_indices: dict = None,
    tol: float = 0.01,
) -> tuple[list[int], list[int]]:
    """Generate a random scaffold train/test split from the given molecules.

    Molecules of the same Bemis-Murcko scaffold are always placed in the same
    split. Random splits of the scaffolds are generated until the resulting
    train proportion is within ``tol`` of the target.

    Args:
        mols (list[Mol]): List of RDKit mol objects.
        test_frac (float): The proportion of the dataset that the test set should
            represent. Strictly between 0 and 1.
        seed (int): The random seed.
        scaffold_to_indices (dict[str, list[int]], optional): A dictionary
            containing scaffolds as keys and corresponding molecule indices as
            values. See ``preprocess_scaffold(mols)``. If None, it is computed
            from ``mols``.
        tol (float): The maximum allowed deviation from the target test
            proportion.

    Returns:
        tuple[list[int], list[int]]: A tuple containing two mutually exclusive
            lists: (train_indices, test_indices).

    Raises:
        ValueError: If ``test_frac`` is not strictly between 0 and 1.
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    if scaffold_to_indices is None:
        scaffold_to_indices = preprocess_scaffold(mols)

    # NOTE: random.seed(seed) seeds the global random module
    random.seed(seed)

    train_prop = 1 - test_frac

    train_indices = []
    test_indices = []

    # Until we get a valid split
    # NOTE:
    # The test proportion will always be less than or equal to test_frac
    # Potential infinite loop if no split can produce a valid scaffold split
    # If train_prop <= tol, this will crash
    while abs((len(train_indices) / len(mols)) - train_prop) > tol:
        train_indices = []
        test_indices = []
        shuffled_unique_scaffolds = random.sample(list(scaffold_to_indices.keys()), k=len(scaffold_to_indices))

        while len(train_indices) < (train_prop * len(mols)):
            cur_scaf = shuffled_unique_scaffolds.pop(0)
            train_indices += scaffold_to_indices[cur_scaf]

        for remaining_scaffolds in shuffled_unique_scaffolds:
            test_indices += scaffold_to_indices[remaining_scaffolds]

    assert len(set(train_indices) & set(test_indices)) == 0
    assert len(set(train_indices + test_indices)) == len(mols)

    return train_indices, test_indices


def umap_split(
    mfps: list[ExplicitBitVect],
    test_frac: float,
    seed: int,
    n_clusters: int = 7,
) -> tuple[list[int], list[int]]:
    """Generate a UMAP-based train/test split from the given molecules.

    Embeds the fingerprints in 2D with UMAP, clusters with k-means, and selects
    the cluster whose size is closest to the target size as the test set.

    Args:
        mfps (list[ExplicitBitVect]): Molecular bit vector fingerprints, one per
            molecule.
        test_frac (float): The proportion of the dataset that the test set should
            represent. Strictly between 0 and 1. The test set will be the cluster
            closest in size to this proportion.
        seed (int): The random seed. Passed to k-means as ``random_state`` and
            to UMAP as ``transform_seed``, which currently does not make the
            UMAP embedding (and therefore the split) reproducible.
        n_clusters (int): The number of clusters to use for the k-means
            clustering.

    Returns:
        tuple[list[int], list[int]]: A tuple containing two mutually exclusive
            lists: (train_indices, test_indices).

    Raises:
        ValueError: If ``test_frac`` is not strictly between 0 and 1, or if
            ``n_clusters`` is not between 2 and the number of molecules.
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    if not 2 <= n_clusters <= len(mfps):
        raise ValueError(f"n_clusters must be between 2 and the number of molecules ({len(mfps)}), got {n_clusters}")

    test_size = round(test_frac * len(mfps))

    mfp_umap = umap.UMAP(n_neighbors=15, n_components=2, transform_seed=seed).fit_transform(mfps)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

    cluster_labels = kmeans.fit_predict(mfp_umap)

    # Can be used to save the embeddings
    # embeddings = (mfp_umap[:, 0], mfp_umap[:, 1])

    _cluster_index, counts = np.unique(cluster_labels, return_counts=True)
    difference_list = [abs(test_size - j) for j in counts]

    min_cluster_index = difference_list.index(min(difference_list))

    umap_test_indices = (np.where(cluster_labels == min_cluster_index)[0]).tolist()
    umap_train_indices = (np.where(cluster_labels != min_cluster_index)[0]).tolist()

    assert len(set(umap_train_indices) & set(umap_test_indices)) == 0
    assert len(set(np.concatenate([umap_train_indices, umap_test_indices]))) == len(mfps)

    return umap_train_indices, umap_test_indices


# Code referenced from https://github.com/arunraja-hub/quadmetformer/blob/main/finetuning/preprocessing_tdc_dataset.py line 1159
def spectral_split(
    aff_mat: np.ndarray,
    data_labels: np.ndarray,
    test_frac: float,
    seed: int,
    task_type: str = "classification",
) -> tuple[list[int], list[int]]:
    """Generate a spectral-clustering-based train/test split from an affinity matrix.

    Clusters the molecules with spectral clustering into ``int(1 / test_frac)``
    clusters and ranks the clusters by how closely their fraction of positive
    labels matches that of the whole dataset (the label-balance score). The test
    set starts as the best-ranked cluster, and the next-best clusters are added
    until another would exceed ``test_frac``. All remaining molecules form
    the train set.

    Args:
        aff_mat (np.ndarray): Square, symmetric, non-negative matrix of molecular
            similarities with zeros along the diagonal, where the value at [u, v]
            is the similarity between molecule u and v.
        data_labels (np.ndarray): Label of each molecule, in the same order as
            the rows of ``aff_mat``.
        test_frac (float): The proportion of the dataset that the test set should
            represent. Greater than 0 and at most 0.5, since the number of
            clusters is ``int(1 / test_frac)`` and must be at least 2.
        seed (int): The random seed.
        task_type (str): Either ``"classification"`` (labels are binary) or
            ``"regression"`` (labels are continuous). For ``"regression"``, the
            labels will be binarized (1 if at or above the mean of
            ``data_labels``, otherwise 0) to compute the label-balance score of
            each cluster.

    Returns:
        tuple[list[int], list[int]]: A tuple containing two mutually exclusive
            lists: (train_indices, test_indices).

    Raises:
        ValueError: If ``test_frac`` is not strictly between 0 and 1, if
            ``test_frac`` is above 0.5 (which would leave fewer than 2
            clusters), if ``task_type`` is not recognized, if ``aff_mat`` and
            ``data_labels`` have different lengths, or if ``aff_mat`` is not
            square, or has negative values, or has non-zero values
            along its diagonal.
    """
    
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")


    if np.ndim(aff_mat) != 2 or np.shape(aff_mat)[0] != np.shape(aff_mat)[1]:
        raise ValueError(f"aff_mat must be a square matrix, got shape {np.shape(aff_mat)}")

    if np.min(aff_mat) < 0:
        raise ValueError(f"aff_mat must be non-negative, got a minimum value of {np.min(aff_mat)}")

    if not np.allclose(np.diag(aff_mat), 0):
        raise ValueError(f"aff_mat must have zeros along its diagonal, got a maximum diagonal value of {np.max(np.diag(aff_mat))}")

    
    cluster_count = int(1 / test_frac)

    if cluster_count < 2:
        raise ValueError(
            f"test_frac must be at most 0.5 to get at least 2 clusters, got {test_frac}"
        )

    if task_type not in ("classification", "regression"):
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )

    if len(aff_mat) != len(data_labels):
        raise ValueError(
            f"aff_mat and data_labels must have the same length, got {len(aff_mat)} and {len(data_labels)}"
        )

    random_state = check_random_state(seed)

    if task_type == "regression":
        # Binarize the labels around their mean, so that clusters can be compared by label balance
        data_labels = (np.asarray(data_labels) >= np.mean(data_labels)).astype(int)

    clustering = SpectralClustering(
        n_clusters=cluster_count,
        assign_labels="kmeans",
        n_init=10,
        random_state=random_state,
        affinity="precomputed",
    ).fit(aff_mat)

    cluster_assignments = clustering.labels_
    clusters = []

    for x in range(cluster_count):
        clusters.append(set(np.asarray(cluster_assignments == x).nonzero()[0]))

    # This will be populated by the lb scores of the clusters
    cluster_scores = np.zeros(cluster_count)

    data_dist = np.count_nonzero(data_labels == 1) / len(data_labels)

    for x, c in enumerate(clusters):
        cluster_dist = sum(data_labels[list(c)]) / len(c)

        # Label-balance score
        lb_score = abs(cluster_dist - data_dist)

        cluster_scores[x] = lb_score

    cluster_order = np.argsort(cluster_scores)

    # Begin the test set with the best cluster
    best_cluster_idx = cluster_order[0]
    test = list(clusters[best_cluster_idx])

    # Add next-best clusters...
    cluster_rank = 1

    while True:
        next_cluster_i = cluster_order[cluster_rank]
        next_cluster = clusters[next_cluster_i]

        # ... until the target size would be exceeded
        if len(test) + len(next_cluster) > test_frac * len(data_labels):
            break

        test.extend(list(next_cluster))
        cluster_rank += 1

        # Could also stop here instead. It increases CSO for the BACE dataset at least
        # if len(test) > test_frac * len(data_labels):
        #     break

    train = [x for x in range(len(data_labels)) if x not in set(test)]

    return train, test