import numpy as np
from rdkit.Chem import AllChem
from models import RandomForestModel
from dataset import MoleculeDataset
import joblib
import numpy as np
from rdkit.Chem import AllChem
import joblib
from rdkit.Chem import SDMolSupplier, SDWriter
import utils
import yaml
import argparse

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def arg_parse():
    parser = argparse.ArgumentParser(description='Train a random forest model for drug discovery')
    parser.add_argument('--config', dest='config', type=str, default='./config/test_10K.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

def predict(args):
    model_path = config['output_params']['model_output_path']
    dataset_path = config['data_params']['test_file']
    output_sdf_path = config['output_params']['output_path']
    output_csv_path = config['output_params']['csv_output_path']
    test_size = config['data_params']['test_size']
    train_size = config['data_params']['train_size']
    # Load the trained RF model
    model = joblib.load(model_path)
    test_dataset = MoleculeDataset(dataset_path)
    X_test = np.array(test_dataset.fps)
    y_test = np.array(test_dataset.scores)
    y_test_pred = model.predict(X_test)

    # Write the predictions to a CSV file
    model_type = config['model_params']['model_type'] + '_' + train_size
    utils.write_pred_scores_to_sdf(y_test_pred, dataset_path, output_sdf_path, model_type)
    utils.write_results_csv(output_csv_path, test_dataset, y_test_pred, model_type)
    
    return y_test_pred

if __name__ == '__main__':
    args = arg_parse()
    config = load_config(args.config)
    predict(args)