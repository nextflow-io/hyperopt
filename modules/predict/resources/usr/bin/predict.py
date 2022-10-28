#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


def encode_onehot(x, categories):
    for column, values in categories.items():
        if column in x:
            for v in values:
                x['%s_%s' % (column, v)] = (x[column] == v)
            x = x.drop(columns=[column])

    return x


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='trained model file', required=True)
    parser.add_argument('--data', help='data file', required=True)
    parser.add_argument('--meta', help='metadata file', required=True)

    args = parser.parse_args()

    # load model
    print('loading model')

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # load dataset
    print('loading dataset')

    df = pd.read_csv(args.data, index_col=0, sep='\t')

    with open(args.meta, 'r') as f:
        meta = json.load(f)

    # extract input features
    x = df[meta['feature_names']]
    x = encode_onehot(x, meta['categories'])

    # extract target column
    target = meta['target_names'][0]
    classes = meta['categories'][target]

    y_true = df[target]

    # perform inference
    print('performing inference')

    y_pred = model.predict(x)
    y_pred = [classes[v] for v in y_pred]

    for sample_name, v_pred, v_true in zip(df.index, y_pred, y_true):
        print('%8s: %8s (%8s)' % (sample_name, v_pred, v_true))

    print()
    print('acc: %0.3f' % (accuracy_score(y_true, y_pred)))
