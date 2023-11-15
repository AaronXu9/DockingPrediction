
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

# import dgl

# def mol_to_graph(mol):
#     # Create a graph
#     g = dgl.DGLGraph()
#     g.add_nodes(mol.GetNumAtoms())
    
#     # Add edges
#     for bond in mol.GetBonds():
#         g.add_edges(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
#         g.add_edges(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
    
#     return g

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
    return

# Define a custom dataset class
class MoleculeDataset(Dataset):
    def __init__(self, sdf_file, type='fingerprints'):
        self.mols = []
        self.ids = []
        self.scores = []
        suppl = Chem.SDMolSupplier(sdf_file)
        self.type = type

        for mol in suppl:
            if mol is not None:
                self.mols.append(mol)
                self.scores.append(float(mol.GetProp('Score')))
                self.ids.append(mol.GetProp('full_synton_id'))
        
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
            return self.mols[idx], self.scores[idx]
        else:
            raise ValueError('Invalid type')

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


# Define a message-passing neural network model
class MPNNModel(nn.Module):
    def __init__(self):
        super(MPNNModel, self).__init__()
        self.embedding = nn.Embedding(2048, 128)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1) # modified input shape
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) # add a channel dimension
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.mean(dim=-1).mean(dim=-1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

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
    mpnn_model = MPNNModel()
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


