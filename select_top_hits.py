from rdkit import Chem
import pandas as pd

def select_top(sdf_file, output_file, score_column='RF_predictions', top=100000):
    # Load SDF file
    # sdf_file = 'large_file.sdf'
    suppl = Chem.SDMolSupplier(sdf_file)

    # Extract molecules and scores
    molecules = []
    scores = []
    for mol in suppl:
        if mol is not None:
            molecules.append(mol)
            scores.append(mol.GetProp(score_column))

    # Create a DataFrame
    df = pd.DataFrame({'Molecule': molecules, 'Score': scores})

    # Convert 'Score' to numeric and sort the DataFrame by 'Score' in descending order
    df['Score'] = pd.to_numeric(df[score_column], errors='coerce')
    df_sorted = df.sort_values(score_column, ascending=True)

    # Select the top 100,000 entries
    top_molecules = df_sorted.head(top)['Molecule']

    # Write the top 100,000 molecules to a new SDF file
    writer = Chem.SDWriter(output_file)
    for mol in top_molecules:
        writer.write(mol)
    writer.close()

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Select top hits from a large SDF file')
    parser.add_argument('sdf_file', type=str, help='SDF file')
    parser.add_argument('--output_file', type=str, default='top_100k.sdf', help='Output SDF file')
    parser.add_argument('--score_column', type=str, default='Score', help='Score column name')
    parser.add_argument('--top', type=int, default=100000, help='Number of top hits to select')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    select_top(args.sdf_file, args.output_file, args.score_column, args.top)
    # select_top('../data/top_100k.sdf')