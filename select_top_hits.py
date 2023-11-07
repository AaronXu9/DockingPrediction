from rdkit import Chem
import pandas as pd

def select_top(sdf_file):
    # Load SDF file
    # sdf_file = 'large_file.sdf'
    suppl = Chem.SDMolSupplier(sdf_file)

    # Extract molecules and scores
    molecules = []
    scores = []
    for mol in suppl:
        if mol is not None:
            molecules.append(mol)
            scores.append(mol.GetProp('Score'))

    # Create a DataFrame
    df = pd.DataFrame({'Molecule': molecules, 'Score': scores})

    # Convert 'Score' to numeric and sort the DataFrame by 'Score' in descending order
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df_sorted = df.sort_values('Score', ascending=False)

    # Select the top 100,000 entries
    top_molecules = df_sorted.head(100000)['Molecule']

    # Write the top 100,000 molecules to a new SDF file
    writer = Chem.SDWriter('top_100k.sdf')
    for mol in top_molecules:
        writer.write(mol)
    writer.close()

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Select top hits from a large SDF file')
    parser.add_argument('sdf_file', type=str, help='SDF file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    select_top(args.sdf_file)