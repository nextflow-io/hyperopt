#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Create a synthetic classification dataset')
    parser.add_argument('--n-samples', help='number of samples', type=int, default=100)
    parser.add_argument('--n-features', help='number of features', type=int, default=20)
    parser.add_argument('--n-classes', help='number of classes', type=int, default=2)
    parser.add_argument('--train-size', help='training set proportion', type=float, default=0.8)
    parser.add_argument('--train-data', help='training data file', default='example.train.data.txt')
    parser.add_argument('--train-labels', help='training label file', default='example.train.labels.txt')
    parser.add_argument('--test-data', help='test data file', default='example.test.data.txt')
    parser.add_argument('--test-labels', help='test label file', default='example.test.labels.txt')

    args = parser.parse_args()

    # create synthetic dataset
    x, y = make_blobs(
        args.n_samples,
        args.n_features,
        centers=args.n_classes)

    # initialize class names
    classes = ['class-%02d' % i for i in range(args.n_classes)]
    y = [classes[y_i] for y_i in y]

    # initialize sample names, feature names
    x_samples = ['sample-%08d' % i for i in range(args.n_samples)]
    x_features = ['feature-%06d' % i for i in range(args.n_features)]

    # initialize dataframes
    x = pd.DataFrame(x, index=x_samples, columns=x_features)
    y = pd.DataFrame(y, index=x_samples, columns=['label'])

    # split dataset into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - args.train_size)

    # save datasets
    x_train.to_csv(args.train_data, sep='\t', float_format='%.8f')
    y_train.to_csv(args.train_labels, sep='\t')

    x_test.to_csv(args.test_data, sep='\t', float_format='%.8f')
    y_test.to_csv(args.test_labels, sep='\t')
