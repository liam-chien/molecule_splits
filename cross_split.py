import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def generate_affinity_matrix(
    data_df: pd.DataFrame,
    save_dir: str | Path,
    save_name: str = "tanimoto_affinity_matrix",
    smiles_col: str = "smiles",
    mfp_radius: int = 2,
    mfp_size: int = 2048,
    verbose: bool = True,
) -> Path:
    """Generate and save a molecular Tanimoto similarity affinity matrix.

    Converts each molecule in ``data_df`` to a Morgan fingerprint and computes
    the pairwise Tanimoto similarity between all molecules. The resulting
    square, symmetric matrix contains the Tanimoto similarity between molecules
    at each pair of indices and is saved as a NumPy ``.npy`` file.

    Args:
        data_df (pd.DataFrame): Dataframe containing a column of molecular
            SMILES strings. The column is specified by ``smiles_col``.
        save_dir (str | Path): Save directory for the affinity matrix ``.npy`` file.
        save_name (str): Name of the saved affinity matrix file without the
            ``.npy`` extension.
        smiles_col (str): Name of the column in ``data_df`` containing the
            molecules' SMILES strings.
        mfp_radius (int): Radius of the Morgan fingerprints used to represent
            the molecules. Must be non-negative.
        mfp_size (int): Number of bits in each Morgan fingerprint. Must be
            positive.
        verbose (bool): Whether to print progress information and warnings.

    Returns:
        Path: Path to the saved affinity matrix ``.npy`` file.

    Raises:
        KeyError: If ``smiles_col`` is not a column in ``data_df``.
        ValueError: If ``data_df`` has fewer than two molecules, if ``mfp_radius``
            is negative, if ``mfp_size`` is not positive, or if any SMILES string
            is invalid.
    """


    if smiles_col not in data_df.columns:
        raise KeyError(f"data_df does not contain the column '{smiles_col}'")

    if len(data_df) < 2:
        raise ValueError("data_df must contain at least two molecules")

    if mfp_radius < 0:
        raise ValueError(f"mfp_radius must be non-negative, got {mfp_radius}")

    if mfp_size <= 0:
        raise ValueError(f"mfp_size must be positive, got {mfp_size}")


    if verbose:
        print("WARNING: Generating the affinity matrix will take a very large amout of memory for larger datasets")
        print(f"Preloading MFPs of radius={mfp_radius}, size={mfp_size}")

    # Preload the morgan fingerprints of all the molecules
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=mfp_radius, fpSize=mfp_size)
    mfps = []

    for idx, smiles in enumerate(data_df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)

        # Flag any invalid SMILES
        if mol is None:
            raise ValueError(f"Invalid SMILES in row {idx}: {smiles}")

        mfps.append(gen.GetFingerprint(mol))


    num_mols = len(mfps)
    if verbose:
        print(f"Creating empty {num_mols} x {num_mols} matrix")
    
    affinity_mat = np.full((num_mols, num_mols), np.nan)

    # Fill diagonal with ones, as this is a same-molecule comparison
    np.fill_diagonal(affinity_mat, 1)

    if verbose:
        print(f"Populating the matrix with Tanimoto similarities")
    
    total_comparisons = int((num_mols ** 2 - num_mols) / 2)
    current = 0
    
    for i, mfp1 in enumerate(mfps):
        for j, mfp2 in enumerate(mfps):
            
            # Skip the diagonal and lower triangular entries; the matrix is symmetric.
            if i >= j:
                continue

            if verbose and (current % 100000 == 0 or current == total_comparisons - 1):
                print("\r" + f"{round((current * 100) / total_comparisons, 2)} %\t\tCurrently on {current} / {total_comparisons}\t\t", end="")
            
            tan_sim = DataStructs.TanimotoSimilarity(mfp1, mfp2)
            affinity_mat[i, j] = tan_sim
            affinity_mat[j, i] = tan_sim

            current += 1

    assert not np.isnan(affinity_mat).any()

    if verbose:
        print()
        print(f"Completed! Saving into directory {save_dir}")

    # Save the matrix
    save_path = Path(f"{save_dir}/{save_name}.npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(save_path, affinity_mat)

    if verbose:
        print("Saved")

    return save_path

def aff_mat_cso(
    aff_mat: np.ndarray,
    idxs1: list[int],
    idxs2: list[int],
    verbose: bool = False,
    save_dir: str | Path | None = None,
    save_name: str | None = None,
    save_size: int = 100000,
) -> float:
    """Calculate the cross-split overlap between two groups of molecules.

    Computes the Tanimoto similarity for every pair formed by the molecules in
    ``idxs1`` and ``idxs2`` using a precomputed affinity matrix. The mean of 
    all cross-group similarities is returned, and the similarity values can
    optionally be saved as a ``.csv`` file.

    Args:
        aff_mat (np.ndarray): Square affinity matrix containing pairwise Tanimoto
            similarities between molecules. Molecule indices correspond to the
            row and column indices of the matrix.
        idxs1 (list[int]): Indices of molecules in the first group.
        idxs2 (list[int]): Indices of molecules in the second group.
        verbose (bool): Whether to print progress information and an estimated
            remaining time.
        save_dir (str | Path | None): Directory in which to save the similarity
            values. Similarities are only saved if both ``save_dir`` and
            ``save_name`` are provided.
        save_name (str | None): Base filename for the saved similarity values,
            without the ``.csv`` extension. Similarities are only
            saved if both ``save_dir`` and ``save_name`` are provided.
        save_size (int): Maximum number of similarity values to save if
            ``save_dir`` and ``save_name`` are provided.

    Returns:
        float: Mean Tanimoto similarity across all pairs between ``idxs1`` and
            ``idxs2``.

    Raises:
        ValueError: If either ``idxs1`` or ``idxs2`` is empty.
    """
    num_comps = len(idxs1) * len(idxs2)

    if num_comps == 0:
        raise ValueError("idxs1 and idxs2 must both contain at least one molecule")

    # Pre-allocate memory for the large array
    # TODO: Make this work as a sum if the data doesn't need to be saved
    tan_sims = np.full(num_comps, np.nan)

    start = time.time()
    last_update = 0

    for i, mol1 in enumerate(idxs1):
        for j, mol2 in enumerate(idxs2):

            arr_idx = i * len(idxs2) + j
            
            tan_sims[arr_idx] = aff_mat[mol1, mol2]
    
            if verbose and arr_idx % 10000 == 0:
                
                elapsed = time.time() - start

                if elapsed - last_update >= 1:

                    last_update += elapsed - last_update

                    current_idx = arr_idx + 1
                    prog = current_idx / num_comps
                
                    print(f"\rCompleted {current_idx} / {num_comps}\t\t{prog * 100:.2f}%\t\tEstimated wait: {elapsed / prog - elapsed:.2f} s          ", end = "")

    if verbose:
        print("\nFinished!")

    if save_dir is not None and save_name is not None:

        # Will print properly if verbose
        save_tan_sim(
            tan_sims,
            str(Path(save_dir) / save_name),
            verbose=verbose,
            save_size=save_size,
        )

    return np.mean(tan_sims)

def aff_mat_cso_k_max(
    aff_mat: np.ndarray,
    idxs1: list[int],
    idxs2: list[int],
    k: int,
    verbose: bool = False,
    save_dir: str | Path | None = None,
    save_name: str | None = None,
    save_size: int = 100000,
) -> float:
    """Calculate the mean of the k highest cross-split overlap values.

    
    Computes the Tanimoto similarity for every pair formed by the molecules in
    ``idxs1`` and ``idxs2`` using a precomputed affinity matrix. The k highest
    cross-split similarities are then averaged, and the similarity values can
    optionally be saved for visualization.

    Args:
        aff_mat (np.ndarray): Square affinity matrix containing pairwise Tanimoto
            similarities between molecules. Molecule indices correspond to the
            row and column indices of the matrix.
        idxs1 (list[int]): Indices of molecules in the first group.
        idxs2 (list[int]): Indices of molecules in the second group.
        k (int): Number of highest cross-split similarity values to average.
            Must be positive and no greater than the total number of
            cross-split comparisons.
        verbose (bool): Whether to print progress information and an estimated
            remaining time.
        save_dir (str | Path | None): Directory in which to save the similarity
            values. Similarities are only saved if both ``save_dir`` and
            ``save_name`` are provided.
        save_name (str | None): Base filename for the saved similarity values,
            without the ``.csv`` extension. Similarities are only
            saved if both ``save_dir`` and ``save_name`` are provided.
        save_size (int): Maximum number of similarity values to save if
            ``save_dir`` and ``save_name`` are provided.

    Returns:
        float: Mean of the k highest Tanimoto similarities across all pairs
            between ``idxs1`` and ``idxs2``.

    Raises:
        ValueError: If either ``idxs1`` or ``idxs2`` is empty, if ``k`` is not
            positive, or if ``k`` exceeds the number of cross-split comparisons.
    """
    
    num_comps = len(idxs1) * len(idxs2)

    if num_comps == 0:
        raise ValueError("idxs1 and idxs2 must both contain at least one molecule")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if k > num_comps:
        raise ValueError(f"k cannot exceed the number of cross-split comparisons ({num_comps}), got {k}")

    # Pre-allocate memory for the large array
    tan_sims = np.full(num_comps, np.nan)

    start = time.time()
    last_update = start

    for i, mol1 in enumerate(idxs1):
        for j, mol2 in enumerate(idxs2):

            arr_idx = i * len(idxs2) + j

            tan_sims[arr_idx] = aff_mat[mol1, mol2]

            if verbose and arr_idx % 10000 == 0:

                current_time = time.time()

                if current_time - last_update >= 1:

                    last_update = current_time

                    elapsed = current_time - start
                    current_idx = arr_idx + 1
                    prog = current_idx / num_comps

                    print(f"\rCompleted {current_idx} / {num_comps}\t\t{prog * 100:.2f} %\t\tEstimated wait: {elapsed / prog - elapsed:.2f} s          ", end = "")

    if verbose:
        print("\nFinished!")

    if save_dir is not None and save_name is not None:
        save_tan_sim(
            tan_sims,
            str(Path(save_dir) / save_name),
            verbose=verbose,
            save_size=save_size,
        )

    # Find and average the k highest cross-split similarities
    max_k_indices = np.argpartition(tan_sims, -k)[-k:]
    return np.mean(tan_sims[max_k_indices])

def save_tan_sim(
    tan_sims: np.ndarray,
    save_path: str | Path,
    verbose: bool = False,
    save_size: int = 100000,
) -> None:
    """Save Tanimoto similarity values of a split to a CSV file.

    Saves all similarity values unless their number exceeds ``save_size``, in
    which case a random sample of ``save_size`` values is saved to reduce the
    size of the output file.

    Args:
        tan_sims (np.ndarray): Array of Tanimoto similarity values to save.
        save_path (str | Path): Base path and filename for the output file,
            without the ``".csv"`` extension.
        verbose (bool): Whether to print progress information and the output
            path.
        save_size (int): Maximum number of similarity values to save. Must be
            positive. If ``tan_sims`` contains more values than ``save_size``,
            a random sample of ``save_size`` values is saved.

    Returns:
        None

    Raises:
        ValueError: If ``save_size`` is not positive.
    """

    save_path = Path(str(save_path) + ".csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_size <= 0:
        raise ValueError(f"save_size must be positive, got {save_size}")

    if verbose:
        print("Saving...")
        
    if len(tan_sims) > save_size:

        if verbose:
            print("Randomly sampling data")

        # TODO: add a seed for this
        save_arr = np.random.choice(tan_sims, save_size, replace=False)
        np.savetxt(save_path, save_arr, delimiter=",", header="similarity", comments="")
        
        if verbose:
            print(f"Saved at {save_path}")
    else:
        np.savetxt(save_path, tan_sims, delimiter=",", header="similarity", comments="")

        if verbose:
            print(f"Saved at {save_path}")

# OLD CODE. Wrote for verification purposes. Should I leave this in? Would it even be useful in the package?

# Converts two molecules to their morgan fingerprints and then computes tanimoto similarity between these two vectors
# Does not use preprocessed affinity matrix... good for verfication
def _tanimoto_similarity(mol1, mol2, radius = 2, fp_size = 2048, gen = None):

    if gen == None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = fp_size)

    mfp1 = mol1
    # String --> mol --> MFP
    if isinstance(mol1, str):
        mfp1 = gen.GetFingerprint(Chem.MolFromSmiles(mol1))
    # mol --> MFP
    elif isinstance(mol1, Chem.Mol):
        mfp1 = gen.GetFingerprint(mol1) 

    mfp2 = mol2
    # String --> mol --> MFP
    if isinstance(mol2, str):
        mfp2 = gen.GetFingerprint(Chem.MolFromSmiles(mol2))
    # mol --> MFP
    elif isinstance(mol2, Chem.Mol):
        mfp2 = gen.GetFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(mfp1, mfp2)

# A simple test to somewhat verify an affinity matrix
def _test_aff_mat_random(aff_mat, source_df):
    
    # Ensure the matrix is up to par
    print("Random testing affinity matrix...")
    
    N = 10000
    
    for i, j in [(random.randint(0, len(source_df) - 1), random.randint(0, len(source_df) - 1)) for _ in range(N)]:
    
        mol1 = i
        mol2 = j
        
        # Affinity matrix method
        aff1 = aff_mat[mol1, mol2]
        aff2 = aff_mat[mol2, mol1]
        
        # Manual method
        man1 = _tanimoto_similarity(source_df.at[mol1, "smiles"], source_df.at[mol2, "smiles"])
        man2 = _tanimoto_similarity(source_df.at[mol2, "smiles"], source_df.at[mol1, "smiles"])
        
        assert aff1 == aff2
        assert aff1 == man1
        assert man1 == man2
        
    print("Test passed!")