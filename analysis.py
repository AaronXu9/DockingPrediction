import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import r2_score, explained_variance_score


def compute_r2(df_combined, y_true, y_pred):
    # Iterate over each column starting from the 3rd column
    for column in df_combined.columns[2:]:
        # Compute R^2 score
        r2 = r2_score(df_combined['Score'], df_combined[column])

        # Compute Q^2 score
        q2 = explained_variance_score(df_combined['Score'], df_combined[column])

        print(f"Column: {column}")
        print(f"R^2 Score: {r2}")
        print(f"Q^2 Score: {q2}")
        print()
    

def compute_hit_percentage(df_combined, n, prediction_column):
    n = 1000
    top_n_by_score = df_combined.sort_values(by='Score', ascending=True).head(n)

    prediction_columns = ['aoxu_nnetregCG_100K', 'aoxu_nnetregCG_1M', 'aoxu_nnetregCG_500K']
    top_n_by_predictions = {col: df_combined.sort_values(by=col, ascending=True).head(n) for col in prediction_columns}

    top_n_by_predictions['aoxu_nnetregCG_500K']['aoxu_nnetregCG_500K']

    true_top_n_molids = set(top_n_by_score['molid'])

    intersections = {col: true_top_n_molids.intersection(set(top_n_df['molid'])) 
                    for col, top_n_df in top_n_by_predictions.items()}

    for col, intersection in intersections.items():
        print(f"Intersection for {col}: {intersection}")
        print(f"Number of molecules in intersection for {col}: {len(intersection)}")

    return intersections

def plot_pairplot(df, column, plot_type='pairplot'):
    # Set the style of the plot
    sns.set(style="ticks")
    if plot_type == 'pairplot':
        pair_plot = sns.pairplot(df, vars=[f'{column}', 'Score'],diag_kind='hist', plot_kws={'s': 5})  # 's' controls the size of the scatter points
        # Set the plot title
        pair_plot.fig.suptitle('Pair Plot of Score and Score_predict', size=15, y=1.02)
    elif plot_type == 'jointplot':
        pair_plot = sns.jointplot(x=f'{column}', y='Score', data=df, kind='reg', height=5)
        pair_plot.fig.suptitle('Joint Plot of Score and Score_predict', size=15, y=1.02)
    
    # Save the plot
    plt.savefig(f'../analysis/{column}_{plot_type}.png')
    # plt.show()

def select_top_n(df, column, n):
    # Sort the DataFrame based on the column
    df_sorted = df.sort_values(by=column, ascending=True)
    # Select the top n rows
    df_top_n = df_sorted.iloc[:n, :]
    return df_top_n

def plot_hist_score(df, n):
    top_score_from_each_file = []
    for column in df.columns[1:]:
        # Sort the DataFrame based on the column
        df_sorted = df.sort_values(by=column, ascending=True)
        # Select the top n rows
        df_top_n = df_sorted.iloc[:n, :]
        # Append the top score to the list
        top_score_from_each_file.append(df_top_n['Score'].values)
    sns.histplot(top_score_from_each_file)
    
