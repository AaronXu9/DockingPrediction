from rdkit import Chem
from rdkit.Chem import AllChem
import h5py
import numpy as np


def compute_fingerprints(sdf_file, fptype, h5_file):
    """
    Computes fingerprints for each molecule in an SDF file and saves them into a h5 file.
    
    Args:
    sdf_file (str): Path to the SDF file.
    fptype (str): Type of fingerprint to compute. Valid options are 'morgan' and 'rdkit'.
    h5_file (str): Path to the h5 file to save the fingerprints.
    """
    with h5py.File(h5_file, 'w') as f:
        for name, fp, score in compute_fingerprints_helper(sdf_file, fptype):
            f.create_dataset(name, data=np.array(fp))
            f[name].attrs['Score'] = score
            
def compute_fingerprints_helper(sdf_file, fptype):
    """
    Computes fingerprints for each molecule in an SDF file.
    
    Args:
    sdf_file (str): Path to the SDF file.
    fptype (str): Type of fingerprint to compute. Valid options are 'morgan' and 'rdkit'.
    
    Yields:
    tuple: A tuple containing the molecule name, its corresponding fingerprint, and its score.
    """
    with open(sdf_file, 'rb') as f:
        suppl = Chem.ForwardSDMolSupplier(f)
        for mol in suppl:
            if mol is not None:
                if mol.HasProp('molid'):
                    name = mol.GetProp("molid")
                else: 
                    name = mol.GetProp('full_synton_id')
                score = float(mol.GetProp('Score'))
                if fptype == 'morgan':
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                elif fptype == 'rdkit':
                    fp = Chem.RDKFingerprint(mol)
                elif fptype == 'maccs':
                    fp = Chem.MACCSkeys.GenMACCSKeys(mol)
                else:
                    raise ValueError('Invalid fingerprint type')
                yield (name, fp, score)

def test_compute_fingerprints():
    # Generate fingerprints
    compute_fingerprints('../data/D2_7jvr_dop_393b_2comp_final_10M_train_1K_2d_score.sdf', 'morgan', '../data/train_1K_fingerprints.h5')
    
    # Load fingerprints
    with h5py.File('../data/train_1K_fingerprints.h5', 'r') as f:
        for name in f:
            fp = np.array(f[name])
            score = f[name].attrs['Score']
            assert isinstance(fp, np.ndarray)
            assert isinstance(score, float)


if __name__ == '__main__':
    # compute_fingerprints('../data/D2_7jvr_dop_393b_2comp_final_10M_train_1K_2d_score.sdf', 'morgan', '../data/train_1K_fingerprints.h5')
    test_compute_fingerprints()
