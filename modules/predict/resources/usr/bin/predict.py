#!/usr/bin/env python3

import argparse
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='trained model file', required=True)
    parser.add_argument('--data', help='data file', required=True)
    parser.add_argument('--labels', help='label file', required=True)

    args = parser.parse_args()

    # load model
    print('loading model')

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # load dataset
    print('loading dataset')

    x = pd.read_csv(args.data, index_col=0, sep='\t')
    y = pd.read_csv(args.labels, index_col=0, sep='\t')

    # select target column
    y.target = y.get(y.columns[0])

    # encode labels
    classes = sorted(set(y.target))

    # perform inference
    print('performing inference')

    y_pred = model.predict(x)
    y_pred = [classes[label] for label in y_pred]

    for sample_name, label_pred, label_true in zip(y.index, y_pred, y.target):
        print('%s: %8s (%8s)' % (sample_name, label_pred, label_true))

    print()
    print('acc: %0.3f' % (accuracy_score(y, y_pred)))
