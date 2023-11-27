# Load the trained model
import numpy as np
from rdkit.Chem import AllChem
import torch
from models import RandomForestModel
from dataset import MoleculeDataset
import joblib
import numpy as np
from rdkit.Chem import AllChem
import joblib
from rdkit.Chem import SDMolSupplier, SDWriter
import utils
from args import arg_parse, load_config
import os
import random
from predict import predict
from rdkit import Chem

def acquire(sample_size, model, train_dataset, train_dataset_scores, new_dataset, new_dataset_scores):
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


def main():
    args = arg_parse()
    config = load_config(args.config)
    # Accessing parameters
    model_path = config['model_params']['model_path']
    dataset_path = config['data_params']['test_file']
    test_size = config['data_params']['test_size']

    # Load the trained RF model
    # model = joblib.load(model_path)
    test_dataset = MoleculeDataset(dataset_path)
    X_test = np.array(test_dataset.fps)
    y_test = np.array(test_dataset.scores)
    model = RandomForestModel()
    model.fit(X_test, y_test)
    y_test_pred = model.predict(X_test)
    
    iters = 5
    exclude_idx = []
    for i in range(5):
        # sort the test dataset by the predicted scores
        top_n_idx = test_dataset.select_top_n_idx(y_test_pred, 100, exclude_indices=exclude_idx)
        
        # train the model on the new dataset
        model.fit(X_test[top_n_idx], y_test[top_n_idx])
        y_test_pred = model.predict(X_test)
        exclude_idx = top_n_idx
        print('iteration: {}, top_n_idx: {}'.format(i, top_n_idx))

if __name__ == '__main__':
    main()