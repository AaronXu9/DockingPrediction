import numpy as np
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from models import RandomForestModel
from dataset import MoleculeDataset
import joblib
import numpy as np
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from models import RandomForestModel
from dataset import MoleculeDataset
import joblib
from rdkit.Chem import SDMolSupplier, SDWriter
from sklearn.metrics import mean_squared_error, r2_score
import argparse

import yaml
import utils
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def arg_parse():
    parser = argparse.ArgumentParser(description='Train a random forest model for drug discovery')
    parser.add_argument('--config', dest='config', type=str, default='./config/config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    config = load_config(args.config)

    # Accessing parameters
    learning_rate = config['model_params']['learning_rate']
    batch_size = config['model_params']['batch_size']
    train_file = config['data_params']['train_file']
    val_file = config['data_params']['val_file']
    model_type = config['model_params']['model_type']
    model_output_path = config['output_params']['model_output_path']
    # ... and so on for other parameters

    train_dataset = MoleculeDataset(train_file)
    val_dataset = MoleculeDataset(val_file)
    # train_loader = DataLoader(train_dataset, batch_size=32,  shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the random forest model
    rf_model = RandomForestModel()
    X_train = np.array(train_dataset.fps)
    y_train = np.array(train_dataset.scores)
    if os.path.exists(model_output_path):
        rf_model = joblib.load(model_output_path)
    else:
        rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)

    # Get predictions of the docking scores on the validation set
    X_val = np.array(val_dataset.fps)
    y_val_pred = rf_model.predict(X_val)

    # Compute R square and Q square for validation set
    r2_train = r2_score(y_train, y_train_pred)
    q2_train = 1 - mean_squared_error(y_train, y_train_pred) / np.var(y_train)

    r2_val = r2_score(val_dataset.scores, y_val_pred)
    q2_val = 1 - mean_squared_error(val_dataset.scores, y_val_pred) / np.var(val_dataset.scores)

    # Save the model to a file
    joblib.dump(rf_model, model_output_path)

    # write the predictions
    utils.write_pred_scores_to_sdf(y_train_pred, train_file, f'train_{int(len(train_dataset) / 1000)}K_predictions.sdf', model_type)
    utils.write_pred_scores_to_sdf(y_val_pred, val_file, f'val_{int(len(val_dataset) / 1000)}K_predictions.sdf', model_type)

    # write the results in csv as well 
    csv_path = f"./results/train_{int(len(train_dataset) / 1000)}K_results.csv"
    utils.write_results_csv(csv_path, train_dataset, y_train_pred, model_type)
    csv_path = f"./results/val_{int(len(val_dataset) / 1000)}K_results.csv"
    utils.write_results_csv(csv_path, val_dataset , y_val_pred, model_type)




