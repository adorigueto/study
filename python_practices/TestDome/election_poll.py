# %% Load necessary libraries
import os
import pandas as pd

# %% Load the election poll dataset from a CSV file
os.chdir(r'C:/Users/Andre/OneDrive/04_Biblioteca/study/python_practices/TestDome')
filename = 'electionpoll.csv'

df = pd.read_csv(filename)
print(df)

# %% Perform calculations
agg_workers_party_votes = df["Workers' Party"].agg({'mean', 'std', 'max', 'min'}).round(1)
print(agg_workers_party_votes)

median_workers_party_votes = int(df["Workers' Party"].median())
print(median_workers_party_votes)

max_votes = df.iloc[:, 1:].max().sort_values(ascending=False)
print(max_votes)

max_votes_day = df.loc[df[max_votes.index[0]] == max_votes[0], 'Date \ Party'].values[0]
print(max_votes_day)

diffs = df.iloc[:, 1:].max() - df.iloc[:, 1:].min()
largest_diff = diffs.max()
largest_diff_col = diffs.idxmax()
print(f"Largest difference: {largest_diff} in column: {largest_diff_col}")