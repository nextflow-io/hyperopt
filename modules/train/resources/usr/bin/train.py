#!/usr/bin/env python3

import argparse
import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='training data file', required=True)
    parser.add_argument('--labels', help='training label file', required=True)
    parser.add_argument('--scaler', help='preprocessing transform to apply to inputs', choices=['maxabs', 'minmax', 'standard'], default='standard')
    parser.add_argument('--model-type', help='which model to train', choices=['dummy', 'gb', 'lr', 'mlp', 'rf'], default='dummy')
    parser.add_argument('--model-name', help='name of trained model file', required=True)

    args = parser.parse_args()

    # load dataset
    print('loading dataset')

    x = pd.read_csv(args.data, index_col=0, sep='\t')
    y = pd.read_csv(args.labels, index_col=0, sep='\t')

    # select target column
    y.target = y.get(y.columns[0])

    # encode labels
    classes = {label: idx for idx, label in enumerate(sorted(set(y.target)))}
    y = [classes[label] for label in y.target]

    # select scaler
    Scaler = {
        'maxabs': MaxAbsScaler,
        'minmax': MinMaxScaler,
        'standard': StandardScaler
    }[args.scaler]

    # select classifier
    Classifier = {
        'dummy': DummyClassifier,
        'gb': GradientBoostingClassifier,
        'lr': LogisticRegression,
        'mlp': MLPClassifier,
        'rf': RandomForestClassifier
    }[args.model_type]

    # create model pipeline
    model = Pipeline([
        ('scaler', Scaler()),
        ('clf', Classifier())
    ])

    # train and evaluate model
    print('training model')

    y_pred = cross_val_predict(model, x, y, cv=5)

    print('mse: %0.3f' % (mean_squared_error(y, y_pred)))
    print('acc: %0.3f' % (accuracy_score(y, y_pred)))

    # train model on full dataset
    model.fit(x, y)

    # save trained model to file
    print('saving model')

    with open(args.model_name, 'wb') as f:
        pickle.dump(model, f)
