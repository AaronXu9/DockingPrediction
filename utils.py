import csv
from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit.Chem import SDMolSupplier, SDWriter
import os

def write_results_csv(file_path, dataset, preds, model_type):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df[f'{model_type}_pred'] = preds
        df.to_csv(file_path, index=False)
    else:
        pd.DataFrame({'molid': dataset.ids, 'score': dataset.scores, f'{model_type}_pred': preds}).to_csv(file_path, index=False)

def write_pred_scores_to_sdf(predictions, orig_sdf_filepath, new_sdf_filepath, model_type):
    # Determine if new_sdf_filepath exists
    sdf_to_read = new_sdf_filepath if os.path.exists(new_sdf_filepath) else orig_sdf_filepath

    # Read the SDF file
    suppl = Chem.SDMolSupplier(sdf_to_read)
    mols = [mol for mol in suppl if mol is not None]

    # Add predictions
    for mol, pred in zip(mols, predictions):
        mol.SetProp(f"{model_type}_Pred", str(pred))

    # Write to new_sdf_filepath
    writer = Chem.SDWriter(new_sdf_filepath)
    for mol in mols:
        writer.write(mol)
    writer.close()