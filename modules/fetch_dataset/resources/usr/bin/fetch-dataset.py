#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.datasets import fetch_openml


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Download an OpenML dataset')
    parser.add_argument('--name', help='dataset name', required=True)
    parser.add_argument('--data', help='data file', default='data.txt')
    parser.add_argument('--labels', help='label file', default='labels.txt')

    args = parser.parse_args()

    # download dataset from openml
    data = fetch_openml(args.name)

    # initialize dataframes
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=data.target_names)

    # save datasets
    x.to_csv(args.data, sep='\t', float_format='%.8f')
    y.to_csv(args.labels, sep='\t')
