#!/usr/bin/env python3

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


def encode_onehot(x, categories):
    for column, values in categories.items():
        if column in x:
            for v in values:
                x['%s_%s' % (column, v)] = (x[column] == v)
            x = x.drop(columns=[column])

    return x


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize a dataset with t-SNE')
    parser.add_argument('--data', help='data file', required=True)
    parser.add_argument('--meta', help='metadata file', required=True)
    parser.add_argument('--outfile', help='output plot file', required=True)

    args = parser.parse_args()

    # load dataset
    df = pd.read_csv(args.data, index_col=0, sep='\t')

    with open(args.meta, 'r') as f:
        meta = json.load(f)

    # extract input features
    x = df[meta['feature_names']]
    x = encode_onehot(x, meta['categories'])

    # extract target column
    target = meta['target_names'][0]
    y = df[target]

    # compute t-SNE embedding
    x_tsne = TSNE().fit_transform(x)

    # plot t-SNE embedding with class labels or colorbar
    plt.axis('off')

    if target in meta['categories']:
        classes = meta['categories'][target]

        for c in classes:
            indices = (y == c)
            plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], label=c, edgecolors='w')

        plt.subplots_adjust(right=0.70)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    else:
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, edgecolors='w')
        plt.colorbar()

    plt.savefig(args.outfile)
    plt.close()
