#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize a dataset with t-SNE')
    parser.add_argument('--data', help='training data file', required=True)
    parser.add_argument('--labels', help='training label file', required=True)
    parser.add_argument('--outfile', help='output plot file', required=True)

    args = parser.parse_args()

    # load dataset
    x = pd.read_csv(args.data, index_col=0, sep='\t')
    y = pd.read_csv(args.labels, index_col=0, sep='\t')

    classes = sorted(set(y.label))

    # compute t-SNE embedding
    x_tsne = TSNE().fit_transform(x)

    # plot t-SNE embedding with class labels
    plt.axis('off')

    for c in classes:
        indices = (y.label == c)
        plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], label=c, edgecolors='w')

    plt.subplots_adjust(right=0.70)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(args.outfile)
    plt.close()
