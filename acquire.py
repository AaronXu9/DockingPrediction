import numpy as np
from rdkit.Chem import AllChem
import torch
from models import RandomForestModel, xgbModel
from dataset import MoleculeDataset
import joblib
from rdkit.Chem import SDMolSupplier, SDWriter
import utils
from args import arg_parse, load_config
import os
import random
from rdkit import Chem
from sklearn.metrics import mean_squared_error, r2_score

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

def train():
    args = arg_parse()
    config = load_config(args.config)

    # Accessing parameters
    learning_rate = config['model_params']['learning_rate']
    batch_size = config['model_params']['batch_size']
    train_file = config['data_params']['train_file']
    val_file = config['data_params']['val_file']
    train_size = config['data_params']['train_size']
    model_type = config['model_params']['model_type'] + '_' + train_size
    model_output_path = config['output_params']['model_output_path']
    # ... and so on for other parameters  

    train_dataset = MoleculeDataset(train_file)
    val_dataset = MoleculeDataset(val_file)

    # Train the random forest model
    model = xgbModel()
    X_train = np.array(train_dataset.fps)
    y_train = np.array(train_dataset.scores)
    # Get predictions of the docking scores on the validation set
    X_val = np.array(val_dataset.fps)

    iters = 5
    exclude_idx = []
    sample_size = int(0.1 * len(train_dataset))

    # Train the model using active learning
    for i in range(iters):
        top_n_idx = train_dataset.subsample_indices(sample_size, exclude_indices=exclude_idx)
        # train the model on the the sampled dataset
        model.fit(X_train[top_n_idx], y_train[top_n_idx])
        # predict the docking scores on rest of the training dataset
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        # sort the test dataset by the predicted scores
        top_n_idx = train_dataset.select_top_n_idx(y_train_pred, sample_size, exclude_indices=exclude_idx)
        
        exclude_idx += top_n_idx
        # print('iteration: {}, top_n_idx: {}'.format(i, top_n_idx))
        #     
        # Compute R square and Q square for validation set
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        q2_train = 1 - mse_train / np.var(y_train)

        mse_val = mean_squared_error(val_dataset.scores, y_val_pred)
        r2_val = r2_score(val_dataset.scores, y_val_pred)
        q2_val = 1 - mse_val / np.var(val_dataset.scores)
        
        print(f'iteration: {i})')
        print(f"mse score on train set: {mse_train}", f"Q2 score on train set: {q2_train}")
        print(f"mse score on val set: {mse_val}", f"Q2 score on val set: {q2_val}")
    
    # train the model on the same dataset sampled from active learning
    model.fit(X_train[exclude_idx], y_train[exclude_idx])
    # predict the docking scores on rest of the training dataset
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    q2_train = 1 - mse_train / np.var(y_train)

    mse_val = mean_squared_error(val_dataset.scores, y_val_pred)
    r2_val = r2_score(val_dataset.scores, y_val_pred)
    q2_val = 1 - mse_val / np.var(val_dataset.scores)
    
    print(f'Overall')
    print(f"mse score on train set: {mse_train}", f"Q2 score on train set: {q2_train}")
    print(f"mse score on val set: {mse_val}", f"Q2 score on val set: {q2_val}")
    
    # Save the model to a file
    joblib.dump(model, model_output_path) 

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
    for i in range(iters):
        # sort the test dataset by the predicted scores
        top_n_idx = test_dataset.select_top_n_idx(y_test_pred, 100, exclude_indices=exclude_idx)
        
        # train the model on the new dataset
        model.fit(X_test[top_n_idx], y_test[top_n_idx])
        y_test_pred = model.predict(X_test)
        exclude_idx = top_n_idx
        print('iteration: {}, top_n_idx: {}'.format(i, top_n_idx))
    
    # save the new dataset
    new_dataset = test_dataset.mols[top_n_idx]
    new_dataset_scores = y_test[top_n_idx]
    new_dataset_sdf_file = '../data/new_dataset.sdf'
    with Chem.SDWriter(new_dataset_sdf_file) as writer:
        for i in range(len(new_dataset)):
            mol = new_dataset[i]
            score = new_dataset_scores[i]
            mol.SetProp('score', str(score))
            mol.SetProp('_Name', str(i))
            writer.write(mol)
    
    # save the model
    prev_model_name = model_path.split('/')[-1].split('.')[0]
    joblib.dump(model, f'../data/{prev_model_name}_AL_{iter}.pkl')

if __name__ == '__main__':
    train()