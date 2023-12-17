from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
import xgboost as xgb
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl.function as fn
from rdkit import Chem
import torch

class GraphNetwork(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphNetwork, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.fc(hg)

# Define a random forest model
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def cv_fit(self, X, y, val_set=None, cv=5):
        # scores = cross_val_score(self.model, X, y, cv=cv)
        self.model.fit(X, y)
        return self.model

class xgbModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def cv_fit(self, X, y, val_set=None, cv=5):
        # scores = cross_val_score(self.model, X, y, cv=cv)
        self.model.fit(X, y)
        return self.model

class MPNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MPNNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, g, feature):
        # Message passing: Aggregate neighbor features
        g.ndata['h'] = feature
        g.update_all(message_func=fn.copy_u('h', 'm'),
                     reduce_func=fn.sum('m', 'h_neigh'))
        h_neigh = g.ndata['h_neigh']
        
        # Node update function: Combine current features with aggregated neighbor features
        h = self.linear(h_neigh)
        return h

class MPNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MPNNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, g, feature):
        # Message passing: Aggregate neighbor features
        g.ndata['h'] = feature
        g.update_all(message_func=fn.copy_u('h', 'm'),
                     reduce_func=fn.sum('m', 'h_neigh'))
        h_neigh = g.ndata['h_neigh']
        
        # Node update function: Combine current features with aggregated neighbor features
        h = self.linear(h_neigh)
        return h

class MPNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MPNN, self).__init__()
        self.layer1 = MPNNLayer(in_dim, hidden_dim)
        self.layer2 = MPNNLayer(hidden_dim, out_dim)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = F.relu(h)
        h = self.layer2(g, h)
        return h

if __name__ == '__main__':
    input_file = '../data/train_1K.sdf'
    from dataset import MoleculeDataset
    dataset = MoleculeDataset(input_file, feat_type='graphs')
    model = GraphNetwork(in_feats=10, hidden_size=128, num_classes=1)  # Modify in_feats and num_classes based on your data
    out = model(dataset.graphs[0])
    print(out)
    molecules = []