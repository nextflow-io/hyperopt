#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Split a dataset into train/test sets')
    parser.add_argument('--data', help='data file', required=True)
    parser.add_argument('--train-size', help='training set proportion', type=float, default=0.8)
    parser.add_argument('--train-data', help='training data file', default='train.txt')
    parser.add_argument('--test-data', help='test data file', default='test.txt')

    args = parser.parse_args()

    # load dataset
    df = pd.read_csv(args.data, index_col=0, sep='\t')

    # split dataset into train/test sets
    df_train, df_test = train_test_split(df, test_size=1 - args.train_size)

    # save datasets
    df_train.to_csv(args.train_data, sep='\t')
    df_test.to_csv(args.test_data, sep='\t')
