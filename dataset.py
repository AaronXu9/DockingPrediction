
# Import necessary libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
import matplotlib.pyplot as plt
from fingerprints import compute_fingerprints
import h5py
import dgl

def mol_to_graph(mol):
    g = dgl.graph()

    # Add nodes with atom features
    for atom in mol.GetAtoms():
        g.add_nodes(1, {'feat': torch.tensor(get_atom_features(atom), dtype=torch.float)})

    # Add edges with bond features
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edges(start, end, {'feat': torch.tensor(get_bond_features(bond), dtype=torch.float)})
        g.add_edges(end, start, {'feat': torch.tensor(get_bond_features(bond), dtype=torch.float)})
    
    return g

def get_atom_features(atom):
    # Example: Use atomic number as a feature
    # You can extend this with more features like atom degree, hybridization, etc.
    return [atom.GetAtomicNum()]

def get_bond_features(bond):
    # Example: Use bond type as a feature
    # Encode bond types as integers (single:1, double:2, triple:3, aromatic:4)
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return [1]
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        return [2]
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        return [3]
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        return [4]
    else:
        return [0]  # Unknown bond type

def atom_features(mol):
    # Example: Use atomic number as feature (other features can be added)
    return torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]).float()

def bond_features(mol):
    # Example: Use bond type as feature
    bond_type = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 4}
    return torch.tensor([bond_type[str(bond.GetBondType())] for bond in mol.GetBonds()]).float()

def extract_features(mols, graphs):
    # Extract features
    for mol, g in zip(mols, graphs):
        g.ndata['feat'] = atom_features(mol)
        # For edges, we duplicate the features since each bond results in two edges (undirected graph)
        g.edata['feat'] = torch.cat([bond_features(mol), bond_features(mol)], dim=0)
    return g

# Define a custom dataset class
class MoleculeDataset(Dataset):
    def __init__(self, sdf_file, feat_type='fingerprints'):
        self.mols = []
        self.ids = []
        self.scores = []
        suppl = Chem.SDMolSupplier(sdf_file)
        self.type = type

        for i, mol in enumerate(suppl):
            if mol is not None:
                self.mols.append(mol)
                self.scores.append(float(mol.GetProp('Score')))
                if mol.HasProp('molid'):
                    self.ids.append(float(mol.GetProp('molid')))
                else:
                    self.ids.append(i)
                    # self.ids.append(mol.GetProp('full_synton_id'))
        
        if feat_type == 'graphs':
            self.graphs = [mol_to_graph(mol) for mol in self.mols]
            # self.graphs = extract_features(self.mols, self.graphs)
        elif feat_type == 'fingerprints':
            if not os.path.exists(f"{sdf_file.split('.sdf')[0]}_fingerprints.h5"):
                compute_fingerprints(sdf_file, 'morgan', f"{sdf_file.split('.sdf')[0]}_fingerprints.h5", )
                # self.fps = np.array(pd.read_hdf(f"{sdf_file.split('.sdf')[0]}_fingerprints.h5"))
                # self.fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2)) for mol in self.mols]
            
            self.fps = []
            with h5py.File(f"{sdf_file.split('.sdf')[0]}_fingerprints.h5", 'r') as f:
                for name in f:
                    fp = np.array(f[name])
                    self.fps.append(fp)
            
            self.fps = np.array(self.fps)
            
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, idx):
        if self.type == 'fingerprints':
            return self.fps[idx], self.scores[idx]
        elif self.type == 'graphs':
            return self.graphs[idx], self.scores[idx]
        else:
            raise ValueError('Invalid type')

    def select_top_n_mol(self, predicted_scores, n):
        """
        Selects top n molecules based on an external array of predicted scores.

        :param predicted_scores: A numpy array containing the predicted scores.
        :param n: The number of top molecules to select.
        :return: A new MoleculeDataset instance with the top n molecules.
        """
        if len(predicted_scores) != len(self.mols):
            raise ValueError("Length of predicted_scores must match the number of molecules in the dataset.")

        # Pair predicted scores with indices
        score_idx_pairs = list(zip(predicted_scores, range(len(predicted_scores))))

        # Sort the pairs based on scores
        sorted_pairs = sorted(score_idx_pairs, key=lambda x: x[0])

        # Select top n indices
        top_n_indices = [idx for _, idx in sorted_pairs[:n]]

        # Create a new dataset with top n molecules
        top_n_dataset = MoleculeDataset.__new__(MoleculeDataset)
        top_n_dataset.mols = [self.mols[i] for i in top_n_indices]
        top_n_dataset.ids = [self.ids[i] for i in top_n_indices]
        top_n_dataset.scores = [self.scores[i] for i in top_n_indices]
        top_n_dataset.fps = np.array([self.fps[i] for i in top_n_indices])
        top_n_dataset.type = self.type

        return top_n_dataset

    def select_top_n_idx(self, predicted_scores, n, exclude_indices=None):
        """
        Selects top n molecules based on an external array of predicted scores.

        :param predicted_scores: A numpy array containing the predicted scores.
        :param n: The number of top molecules to select.
        :return: A new MoleculeDataset instance with the top n molecules.
        """
        if len(predicted_scores) != len(self.mols):
            raise ValueError("Length of predicted_scores must match the number of molecules in the dataset.")

        # Pair predicted scores with indices
        score_idx_pairs = [(score, idx) for idx, score in enumerate(predicted_scores) if idx not in exclude_indices]

        # Sort the pairs based on scores
        sorted_pairs = sorted(score_idx_pairs, key=lambda x: x[0])

        # Select top n indices
        top_n_indices = [idx for _, idx in sorted_pairs[:n]]
        
        return top_n_indices
    
    def subsample_indices(self, sample_size, exclude_indices=None):
        """
        Generates subsample indices. If exclude_indices is provided, it excludes those indices. 
        Otherwise, performs random subsampling.

        :param sample_size: The size of the subsample.
        :param exclude_indices: A list of indices to exclude from the subsampling.
        :return: A list of indices representing the subsample.
        """
        if exclude_indices is None:
            exclude_indices = []
        if len(exclude_indices) + sample_size > len(self.mols):
            raise ValueError("Sample size too large after excluding specified indices.")

        # Generate a set of all indices and remove the excluded indices
        all_indices = set(range(len(self.mols)))
        excluded_set = set(exclude_indices)
        available_indices = list(all_indices - excluded_set)

        # Randomly select indices for subsampling
        subsample_indices = random.sample(available_indices, sample_size)

        return subsample_indices
    
class LargeMoleculeDataset(Dataset):
    def __init__(self, sdf_file):
        self.sdf_file = sdf_file
        self.suppl = Chem.SDMolSupplier(sdf_file)
        self.num_mols = len(self.suppl)
        
    def __len__(self):
        return self.num_mols
    
    def __getitem__(self, idx):
        mol = self.suppl[idx]
        if mol is None:
            raise ValueError(f"Molecule at index {idx} is None")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        score = float(mol.GetProp('Score'))
        return mol, fp, score

# Define a function to train the model
def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.float(), target.float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

# Define a function to evaluate the model
def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target.unsqueeze(1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def sample():
    # Define the subsample size
    subsample_size = 1000

    # Get the indices of the subsample
    subsample_indices = random.sample(range(len(train_dataset)), subsample_size)

    # Create the subsampled dataset
    subsampled_train_dataset = torch.utils.data.Subset(train_dataset, subsample_indices)
    # Save the subsampled dataset in a sdf file
    subsampled_sdf_file = '../data/subsampled_dataset.sdf'
    with Chem.SDWriter(subsampled_sdf_file) as writer:
        for i in range(len(subsampled_train_dataset)):
            mol = train_dataset[i][0]
            fp = train_dataset[i][1]
            score = train_dataset[i][2]
            mol.SetProp('score', str(score))
            mol.SetProp('_Name', str(i))
            writer.write(mol)

    return

if __name__ == '__main__':
    train_dataset = MoleculeDataset('../data/D2_7jvr_dop_393b_2comp_final_10M_train_1K_2d_score.sdf')
    val_dataset = MoleculeDataset('../data/D2_7jvr_dop_393b_2comp_final_10M_test_10K.sdf')
    train_loader = DataLoader(train_dataset, batch_size=32,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the message-passing neural network model
    from models import MPNN
    mpnn_model = MPNN()
    optimizer = optim.Adam(mpnn_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    for epoch in range(10):
        train_loss = train(mpnn_model, train_loader, optimizer, criterion)
        val_loss = evaluate(mpnn_model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

    # Visualize the training process
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()


