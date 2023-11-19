import csv
from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit.Chem import SDMolSupplier, SDWriter
import os
import seaborn as sns
import matplotlib.pyplot as plt

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
        mol.SetProp(f'{model_type}_pred', str(pred))

    # Write to new_sdf_filepath
    writer = Chem.SDWriter(new_sdf_filepath)
    for mol in mols:
        writer.write(mol)
    writer.close()

    import matplotlib.pyplot as plt


def plot_pairplot(df, model_type, set_type):
    # Set the style of the plot
    sns.set(style="ticks")
    pair_plot = sns.pairplot(df, vars=[f'{model_type}_pred', 'score'],diag_kind='hist', plot_kws={'s': 5})  # 's' controls the size of the scatter points
    # Set the plot title
    pair_plot.fig.suptitle('Pair Plot of Score and Score_predict', size=15, y=1.02)
    # Save the plot
    plt.savefig(f'./analysis/{set_type}_{model_type}_pairplot.png')
    # plt.show()


def compute_r2_score():
    return 

def compute_q2_score():
    return 

if __name__ == '__main__':
    plot_pairplot(pd.read_csv('./results/train_10K_results.csv'), 'RF_10K', 'train')
    plot_pairplot(pd.read_csv('./results/train_100K_results.csv'), 'RF_100K', 'train')
    plot_pairplot(pd.read_csv('./results/val_10K_results.csv'), 'RF_10K', 'val')
    plot_pairplot(pd.read_csv('./results/val_10K_results.csv'), 'RF_100K', 'val')