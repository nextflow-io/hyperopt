# hyperopt

A proof-of-concept pipeline for performing hyperparameter optimization of machine learning models with Nextflow.


## Requirements

* Unix-like operating system (Linux, macOS, etc)
* Java >=11
* [Conda](https://docs.conda.io/en/latest/) or [Docker](https://docs.docker.com/)


## Quickstart

1. Install Nextflow (version 22.10.x or higher):
    ```bash
    curl -s https://get.nextflow.io | bash
    ```

2. Launch the pipeline:
    ```bash
    # use conda natively (requires Conda)
    ./nextflow run nextflow-io/hyperopt -profile conda

    # use Wave containers (requires Docker)
    ./nextflow run nextflow-io/hyperopt -profile wave
    ```

3. When the pipeline completes, you can view the training and prediction results in the `results` folder.

Note: the first time you execute the pipeline, Nextflow will take a few minutes to download the pipeline code from this GitHub repository and any related software dependencies (e.g. conda packages or Docker images).


## Configuration

The hyperopt pipeline consists of the following steps:

1. Download a dataset
2. Split the dataset into train/test sets
3. Visualize the train/test sets
4. Train a variety of models on the training set
5. Evaluate each model on the test set
6. Select the best model based on evaluation score

You can control many aspects of this workflow with the pipeline parameters, including:

* Enable/disable each individual step
* Download a different dataset (default is `wdbc`, see [OpenML.org](https://www.openml.org/search?type=data&status=active) to view available datasets)
* Provide your own training data instead of downloading it
* Provide your own pre-trained model and test data
* Select different models (see the `train` module for all available options)

See the `nextflow.config` file for the list of pipeline parameters.


## Cluster support

Since [Nextflow](http://www.nextflow.io) provides an abstraction between the pipeline logic and the underlying execution environment, the hyperopt pipeline can be executed on a single computer or an HPC cluster without any modifications.

Visit the [Nextflow documentation](https://www.nextflow.io/docs/latest/executor.html) to see which HPC schedulers are supported, and how to use them.


## Components

The hyperopt pipeline uses Python (>=3.10) and several Python packages for machine learning and data science. These dependencies are defined in the `conda.yml` file.
