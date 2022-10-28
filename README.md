# ml-hyperopt

A proof-of-concept pipeline for performing hyperparameter optimization of machine learning models with Nextflow.


## Requirements

* Unix-like operating system (Linux, macOS, etc)
* Java 11
* [Anaconda](https://www.anaconda.com/products/distribution) or [Docker](https://docs.docker.com/)


## Quickstart

1. Install Nextflow (version 22.10.x or higher):
    ```bash
    curl -s https://get.nextflow.io | bash
    ```

2. Launch the pipeline:
    ```bash
    # use conda natively (requires Anaconda)
    ./nextflow run nextflow-io/ml-hyperopt -profile conda

    # use Wave containers (requires Docker and Tower Cloud account)
    ./nextflow run nextflow-io/ml-hyperopt -profile wave
    ```

3. When the pipeline completes, you can view the training and prediction results in the `results` folder.

Note: the first time you execute the pipeline, Nextflow will take a few minutes to download the pipeline code from this GitHub repository and any related software dependencies (e.g. conda packages or Docker images).


## Configuration

The ml-hyperopt pipeline consists of the following steps:

1. Download a dataset
2. Split the dataset into train/test sets
3. Visualize the train/test sets
4. Train a variety of models on the train set
5. Evaluate each model on the test set

You can control many aspects of this workflow with the `params` scope of the configuration, including:

* Enable/disable each individual step
* Download a different dataset (default is `iris`, see [OpenML.org](https://www.openml.org/search?type=data&status=active) to view available datasets)
* Provide your own training data instead of downloading it
* Provide your own pre-trained model and test data
* Select different models (see the `train` module for all available options)


## Cluster support

Since [Nextflow](http://www.nextflow.io) provides an abstraction between the pipeline logic and the underlying execution environment, the ml-hyperopt pipeline can be executed on a single computer or an HPC cluster without any modifications.

Visit the [Nextflow documentation](https://www.nextflow.io/docs/latest/executor.html) to see which HPC schedulers are supported, and how to use them.


## Components

The ml-hyperopt pipeline uses Python (>=3.10) and several Python packages for machine learning and data science. These dependencies are defined in the `conda.yml` file.
