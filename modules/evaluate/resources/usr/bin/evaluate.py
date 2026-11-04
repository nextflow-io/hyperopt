#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, r2_score


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
    parser.add_argument('--outfile', help='score file', default='score.json')

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
    y_true = df[target]
    is_categorical = target in meta['categories']

    # perform inference
    print('performing inference')

    y_pred = model.predict(x)

    if is_categorical:
        classes = meta['categories'][target]
        y_pred = [classes[v] for v in y_pred]

    for sample_name, v_pred, v_true in zip(df.index, y_pred, y_true):
        print('%8s: %8s (%8s)' % (sample_name, v_pred, v_true))

    # save score
    if is_categorical:
        score = {
            'name': 'accuracy',
            'value': accuracy_score(y_true, y_pred)
        }

    else:
        score = {
            'name': 'r2',
            'value': r2_score(y_true, y_pred)
        }

    print('%s: %0.3f' % (score['name'], score['value']))

    with open(args.outfile, 'w') as f:
        json.dump(score, f)
