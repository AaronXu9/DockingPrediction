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
import argparse

import yaml

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
    model_output_path = config['model_params']['model_output_path']
    # ... and so on for other parameters

    train_dataset = MoleculeDataset('../data/D2_7jvr_dop_393b_2comp_final_10M_train_100K_2d_score.sdf')
    val_dataset = MoleculeDataset('../data/D2_7jvr_dop_393b_2comp_final_10M_test_10K.sdf')
    # train_loader = DataLoader(train_dataset, batch_size=32,  shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the random forest model
    rf_model = RandomForestModel()
    X_train = np.array(train_dataset.fps)
    y_train = np.array(train_dataset.scores)
    rf_model.fit(X_train, y_train)

    # Get predictions of the docking scores on the validation set
    X_val = np.array(val_dataset.fps)
    y_val_pred = rf_model.predict(X_val)

    # Save the model to a file
    joblib.dump(rf_model, model_output_path)

    # Write predictions to file
    with open('predictions.txt', 'w') as f:
        for pred in y_val_pred:
            f.write(str(pred) + '\n')





