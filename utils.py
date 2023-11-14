import csv
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter

def write_results(results, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['molid', 'score', 'pred_score'])
        for result in results:
            writer.writerow(result)

def write_pred_scores_to_sdf(pred_scores, orig_sdf_filepath, new_sdf_filepath, model_type):
    # Read the original sdf file
    suppl = SDMolSupplier(orig_sdf_filepath)

    # Create a new SDWriter to write the modified molecules
    writer = SDWriter(new_sdf_filepath)

    # Loop through each molecule in the original sdf file
    for mol, pred_score in zip(suppl, pred_scores):
        # Add a new property to the molecule with the predicted score
        mol.SetProp(f'{model_type}_PREDICTED_SCORE', str(pred_score))

        # Write the modified molecule to the new sdf file
        writer.write(mol)

    # Close the SDWriter
    writer.close()