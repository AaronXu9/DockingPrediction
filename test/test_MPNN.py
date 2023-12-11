import torch
import torch.nn.functional as F
import unittest
from dgl import DGLGraph
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models  # Import your model here

def create_mock_graph(num_nodes=5, num_features=3):
    # Create a graph with 'num_nodes' nodes and 'num_features' features per node
    g = DGLGraph()
    g.add_nodes(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Adding edges (excluding self-loops)
                g.add_edges(i, j)
    
    # Random features
    features = torch.rand(num_nodes, num_features)
    
    return g, features

class TestMPNNModel(unittest.TestCase):

    def test_output_shape(self):
        num_nodes = 5
        num_features = 3
        hidden_dim = 10
        out_dim = 2

        # Create a mock graph and model
        g, features = create_mock_graph(num_nodes, num_features)
        model = models.MPNN(num_features, hidden_dim, out_dim)

        # Forward pass
        output = model(g, features)

        # Check output shape
        self.assertEqual(output.shape, (num_nodes, out_dim))

    # You can add more tests, such as checking for output types, handling of edge cases, etc.


if __name__ == '__main__':
    unittest.main()