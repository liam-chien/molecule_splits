import pandas as pd
from pathlib import Path
import json
from rdkit import Chem


# Atomic numbers for organic elements allowed in the cleaned dataset
CHNOPS = {6, 1, 7, 8, 15, 16}

# Atomic numbers for allowed halogens
HALIDES = {9, 17, 35, 53}

# Additional allowed elements
EXTRAS = {5}


def clean_data(
    data_path: str | Path,
    save_dir: str | Path,
    mols_to_exclude: list[str] | None = None,
    verbose: bool = False,
) -> Path:
    """Clean a molecular dataset and save the cleaned data and filtering information.

    SMILES strings are converted to RDKit molecules, canonicalized, and filtered
    for duplicates, invalid molecules, and molecules containing atoms outside the
    allowed element set. Information about each modification or removal is saved
    to a JSON file alongside the cleaned dataset.

    Args:
        data_path (str | Path): Path to the input CSV file. The file must contain a
            ``"smiles"`` column.
        save_dir (str | Path): Directory in which to save the cleaned dataset and
            filtering information.
        mols_to_exclude (list[str] | None): List of canonical SMILES strings to
            remove explicitly. If None, no molecules are explicitly excluded.
        verbose (bool): Whether to print progress while cleaning the dataset.

    Returns:
        Path: Path to the saved cleaned CSV file.

    Raises:
        KeyError: If the input dataframe does not contain a ``"smiles"`` column.
    """

    data_df = pd.read_csv(data_path)

    if "smiles" not in data_df.columns:
        raise KeyError("Input dataframe must contain a \"smiles\" column")

    if mols_to_exclude is None:
        mols_to_exclude = []
    
    # For marking progress
    max_idx = len(data_df) - 1
    
    
    # Start a json that will store which molecules were edited/removed
    # MANUALLY_REMOVED: Specified explicitly to be removed
    # DUPLICATE: Molecule already seen in dataset
    # INVALID: SMILES string not recognized
    # CANONICALIZED: Valid molecule, changed to canon SMILES string
    # FORBIDDEN_ATOMS: Molecule contains disallowed atoms
    info = {"source": str(data_path), "manually_removed" : [], "duplicate" : [], "invalid" : [], "canonicalized" : [], "forbidden_atoms" : []}

    remove_idxs = []
    
    # Canonicalize and filter the dataset
    for idx, smiles in enumerate(data_df["smiles"]):

        if verbose:
            if idx % 100 == 0 or idx == max_idx:
                print("\r" + f"Cleaning index {idx}/{max_idx}", end="")

        mol = Chem.MolFromSmiles(smiles)

        # SMILES is invalid and cannot be converted to an RDKit molecule
        if mol is None:
            info["invalid"].append((idx, smiles))
            remove_idxs.append(idx)
            continue
        
        canonical_smiles = Chem.MolToSmiles(mol)
        
        # If specified to be removed
        if (canonical_smiles in mols_to_exclude):

            info["manually_removed"].append((idx, smiles))
            remove_idxs.append(idx)
            continue

        # If we've already seen this molecule
        elif (canonical_smiles in data_df.loc[0:idx - 1, "smiles"]):

            # TODO: Check for conflicting labels

            info["duplicate"].append((idx, smiles))
            remove_idxs.append(idx)
            continue

        # Check for any exceptional atoms in the molecule
        atoms = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        disallowed_atoms = atoms.difference(CHNOPS | HALIDES | EXTRAS)

        # If there is an Na remaining and there is an [NaH] structure in the SMILES,
        # we will keep this molecule.
        # Many molecules in the MoleculeNet HIV dataset would be invalid without this check
        na_h = (disallowed_atoms == {11}) and ("[NaH]" in smiles)
            
        # Remove molecules containing any disallowed atoms unless the only disallowed atom is Na in [NaH].
        # TODO: Recognize molecules with an [NaH] but also an Na not in an [NaH] as invalid
        if disallowed_atoms and not na_h:
            info["forbidden_atoms"].append((idx, smiles))
            remove_idxs.append(idx)
            continue
        
        # Replace non-canonical SMILES with the RDKit canonical representation
        elif canonical_smiles != smiles:
            info["canonicalized"].append((idx, smiles))
            data_df.at[idx, "smiles"] = canonical_smiles

    data_df.drop(index=remove_idxs, inplace=True)

    if verbose:
        print()
        print(f"Done! Saving data into folder {save_dir}")

    # Save the filter info
    info_save_path = Path(save_dir) / "data_info.json"
    info_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(info_save_path, "w") as f:
        json.dump(info, f, indent=2)    

    # Save the cleaned data
    data_save_path = Path(save_dir) / "clean_data.csv"  
    data_df.to_csv(data_save_path, index = False)

    return data_save_path


# OLD CODE. Probably will remove this

# Make sure a loaded dataframe is all proper smiles
# def check_validity(data_df):
    
#     print("Checking validity...")

#     duplicates = data_df[data_df.duplicated(keep=False)]
    
#     assert len(duplicates) == 0, f"Duplicate found! See df: {duplicates}"
    
#     for smiles in data_df["smiles"]:
#         mol = Chem.MolFromSmiles(smiles)  
#         Chem.MolToSmiles(mol)
    
#     print("If nothing was printed, everything is valid!")
