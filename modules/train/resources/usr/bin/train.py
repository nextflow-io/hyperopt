#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler


def is_categorical(y):
    return y.dtype.kind in 'OSUV'


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
    parser.add_argument('--data', help='training data file', required=True)
    parser.add_argument('--meta', help='training metadata file', required=True)
    parser.add_argument('--scaler', help='preprocessing transform to apply to inputs', choices=['maxabs', 'minmax', 'standard'], default='standard')
    parser.add_argument('--model-type', help='which model to train', choices=['dummy', 'gb', 'lr', 'mlp', 'rf'], default='dummy')
    parser.add_argument('--model-name', help='name of trained model file', default='model.pkl')

    args = parser.parse_args()

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

    if is_categorical(df[target]):
        classes = {v: i for i, v in enumerate(meta['categories'][target])}
        y = df[target].apply(lambda v: classes[v])

    else:
        y = df[target]

    # select scaler
    Scaler = {
        'maxabs': MaxAbsScaler,
        'minmax': MinMaxScaler,
        'standard': StandardScaler
    }[args.scaler]

    # select estimator
    Estimator = {
        True: {
            'dummy': DummyClassifier,
            'gb': GradientBoostingClassifier,
            'lr': LogisticRegression,
            'mlp': MLPClassifier,
            'rf': RandomForestClassifier
        },
        False: {
            'dummy': DummyRegressor,
            'gb': GradientBoostingRegressor,
            'lr': LinearRegression,
            'mlp': MLPRegressor,
            'rf': RandomForestRegressor
        }
    }[is_categorical(df[target])][args.model_type]

    # create model pipeline
    model = Pipeline([
        ('scaler', Scaler()),
        ('estimator', Estimator())
    ])

    # train and evaluate model
    print('training model')

    y_pred = cross_val_predict(model, x, y, cv=5)

    scorers = {
        True: [
            ('mse', mean_squared_error),
            ('mae', mean_absolute_error),
            ('acc', accuracy_score)
        ],
        False: [
            ('mse', mean_squared_error),
            ('mae', mean_absolute_error),
            ('r2', r2_score)
        ]
    }[is_categorical(df[target])]

    for name, score_fn in scorers:
        print('%s: %0.3f' % (name, score_fn(y, y_pred)))

    # train model on full dataset
    model.fit(x, y)

    # save trained model to file
    print('saving model')

    with open(args.model_name, 'wb') as f:
        pickle.dump(model, f)
