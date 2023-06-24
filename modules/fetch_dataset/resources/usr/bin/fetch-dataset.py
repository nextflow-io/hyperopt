#!/usr/bin/env python3

import argparse
import json
from sklearn.datasets import fetch_openml


def is_categorical(y):
    return y.dtype.kind in 'OSUV'


def get_categories(df):
    result = {}
    for c in df.columns:
        if is_categorical(df[c]):
            values = df[c].unique().tolist()

            # fix bug with numerical categories
            if sum(v.isdigit() for v in values) == len(values):
                values = [int(v) for v in values]

            result[c] = values

    return result


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Download an OpenML dataset')
    parser.add_argument('--name', help='dataset name', required=True)
    parser.add_argument('--data', help='data file', default='data.txt')
    parser.add_argument('--meta', help='metadata file', default='meta.json')

    args = parser.parse_args()

    # download dataset from openml
    dataset = fetch_openml(args.name, as_frame=True)

    # save data
    dataset.frame.to_csv(args.data, sep='\t')

    # save metadata
    meta = {
        'name': args.name,
        'feature_names': dataset.feature_names,
        'target_names': dataset.target_names,
        'categories': get_categories(dataset.frame) 
    }

    with open(args.meta, 'w') as f:
        json.dump(meta, f)
