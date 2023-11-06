
import h5py
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem

class FingerprintDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5py.File(h5_file, 'r')
        self.fingerprints = self.h5_file['fingerprints']
        self.mols = self.h5_file['mols']
        

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        fingerprint = np.array(self.fingerprints[idx])
        mol_str = self.mols[idx]
        mol = Chem.MolFromSmiles(mol_str)
        if mol is None:
            raise ValueError(f"Molecule at index {idx} is None")
        AllChem.Compute2DCoords(mol)
        graph = Chem.MolToMolBlock(mol)
        return fingerprint, graph

