#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Split a dataset into train/test sets')
    parser.add_argument('--data', help='data file', required=True)
    parser.add_argument('--labels', help='label file', required=True)
    parser.add_argument('--train-size', help='training set proportion', type=float, default=0.8)
    parser.add_argument('--train-data', help='training data file', default='train.data.txt')
    parser.add_argument('--train-labels', help='training label file', default='train.labels.txt')
    parser.add_argument('--test-data', help='test data file', default='test.data.txt')
    parser.add_argument('--test-labels', help='test label file', default='test.labels.txt')

    args = parser.parse_args()

    # load dataset
    x = pd.read_csv(args.data, index_col=0, sep='\t')
    y = pd.read_csv(args.labels, index_col=0, sep='\t')

    # split dataset into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - args.train_size)

    # save datasets
    x_train.to_csv(args.train_data, sep='\t', float_format='%.8f')
    y_train.to_csv(args.train_labels, sep='\t')

    x_test.to_csv(args.test_data, sep='\t', float_format='%.8f')
    y_test.to_csv(args.test_labels, sep='\t')
