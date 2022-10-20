#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.datasets import make_blobs


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Create a synthetic classification dataset')
    parser.add_argument('--n-samples', help='number of samples', type=int, default=100)
    parser.add_argument('--n-features', help='number of features', type=int, default=20)
    parser.add_argument('--n-classes', help='number of classes', type=int, default=2)
    parser.add_argument('--data', help='data file', default='data.txt')
    parser.add_argument('--labels', help='label file', default='labels.txt')

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

    # save datasets
    x.to_csv(args.data, sep='\t', float_format='%.8f')
    y.to_csv(args.labels, sep='\t')
