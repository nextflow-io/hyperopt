# hyperopt

A proof-of-concept pipeline for performing hyperparameter optimization of machine learning models with Nextflow.


## Requirements

* Unix-like operating system (Linux, macOS, etc)
* Java >=17
* [Conda](https://docs.conda.io/en/latest/) or [Docker](https://docs.docker.com/)


## Quickstart

1. Install Nextflow (version 26.04 or higher):

    ```bash
    curl -s https://get.nextflow.io | bash
    ```

2. Launch the pipeline:

    ```bash
    # use conda natively (requires Conda)
    ./nextflow run nextflow-io/hyperopt -profile test,conda

    # use Wave containers (requires Docker)
    ./nextflow run nextflow-io/hyperopt -profile test,wave
    ```

3. When the pipeline completes, you can view the training and prediction results in the `results` folder.

> [!NOTE]
>
> When you run the pipeline for the first time, it will take a moment to download the pipeline from this GitHub repository and any related software dependencies (e.g. conda packages or Docker images).


## Configuration

The hyperopt pipeline consists of the following steps:

1. Prepare train/test splits from OpenML or user-provided datasets
3. Visualize the train/test sets
4. Train a variety of models on each training set
5. Evaluate each model against each test set
6. Report the best model for each dataset based on evaluation score

You can control many aspects of this workflow with the pipeline parameters, including:

* Download any number of datasets from [OpenML.org](https://www.openml.org/search?type=data&status=active) (default is `wdbc`)
* Evaluate against a number of model types (default is `dummy,gb,lr,mlp,rf`)
* Provide your own train/test splits
* Provide your own pre-trained models

See the `nextflow.config` file for the list of pipeline parameters.


## Executors

Since [Nextflow](http://www.nextflow.io) provides an abstraction between the pipeline logic and the underlying execution environment, the hyperopt pipeline can be executed seamlessly on a local machine, an HPC cluster, or a cloud provider.

See the Nextflow documentation to learn more about [Executors](https://docs.seqera.io/nextflow/executor) and [Configuration](https://docs.seqera.io/nextflow/config).

## Software dependencies

The hyperopt pipeline uses Python (>=3.14) and several Python packages for machine learning and data science. These dependencies are defined in the `conda.yml` file.
