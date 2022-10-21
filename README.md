# ML-Example pipeline

A basic pipeline for training, evaluating, and using machine learning models with Nextflow.


## Requirements

* Unix-like operating system (Linux, macOS, etc)
* Java 11


## Quickstart

1. Install Docker. Read more [here](https://docs.docker.com/).

2. Install Nextflow (version 22.10.x or higher):
    ```bash
    curl -s https://get.nextflow.io | bash
    ```

3. Launch the pipeline:
    ```bash
    ./nextflow run nextflow-io/ml-example -profile conda
    ```

4. When the pipeline completes, you can view the training and prediction results in the `results` folder.

Note: the first time you execute the pipeline, Nextflow will take a few minutes to download the pipeline code from this GitHub repository and any associated Docker images.


## Cluster support

ML-Example execution relies on [Nextflow](http://www.nextflow.io), which provides an abstraction between the pipeline logic and the underlying execution environment. As a result, the pipeline can be executed on a single computer or a HPC cluster without any modifications.

Visit the [Nextflow documentation](https://www.nextflow.io/docs/latest/executor.html) to see which HPC schedulers are supported, and how to use them.


## Components

ML-Example uses Python (>=3.10) and several Python packages for machine learning and data science. These dependencies can be found in the `conda.yml` file.
