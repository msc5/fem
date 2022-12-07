
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict

from rich import print

import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv('results.csv')

    df.iloc[:, 0] = pd.Series(df.iloc[:, 0]).fillna(method='ffill')

    data = defaultdict(dict)
    for i, row in df.iterrows():
        if row.iloc[1] != 'Dump':
            data[row.iloc[0]][row.iloc[1]] = list(row.iloc[2:])

    # print(data[0.1])
    plt.plot(data[0.1]['Stress Error'])
