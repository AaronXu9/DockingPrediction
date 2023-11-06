import numpy as np
from rdkit.Chem import AllChem
from models import RandomForestModel
from dataset import MoleculeDataset
import joblib
import numpy as np
from rdkit.Chem import AllChem
import joblib
from rdkit.Chem import SDMolSupplier, SDWriter

def predict(model_path, dataset_path, output_path='predict.sdf'):
    # Load the trained RF model
    rf_model = joblib.load(model_path)

    # Set up the SDMolSupplier to read the large .sdf file in chunks
    mol_supplier = SDMolSupplier(dataset_path)

    # Set up the SDWriter to write the predicted scores to the .sdf file
    writer = SDWriter(output_path)

    # Loop over the chunks of molecules
    chunk_size = 1000
    for i in range(0, len(mol_supplier), chunk_size):
        # Get the chunk of molecules
        for i in range(0, len(mol_supplier), chunk_size):
            chunk = [mol_supplier[j] for j in range(i, min(i + chunk_size, len(mol_supplier)))]

        # Convert the chunk of molecules to fingerprints
        fps = []
        for mol in chunk:
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
            else:
                fps.append(None)

        # Predict the scores for the fingerprints using the trained RF model
        X = np.array(fps)
        y_pred = rf_model.predict(X)

        # Write the predicted scores to the .sdf file by adding a new column named "RF_predictions"
        for mol, score in zip(chunk, y_pred):
            if mol is not None:
                mol.SetProp('RF_predictions', str(score))
                writer.write(mol)

    # Close the SDWriter
    writer.close()

if __name__ == '__main__':
    predict('100K_model.joblib', '../data/D2_7jvr_dop_393b_2comp_final_10M_train_100K_2d_score.sdf')