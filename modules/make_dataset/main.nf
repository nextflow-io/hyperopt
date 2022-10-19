
process make_dataset {
    publishDir params.outdir, mode: 'copy'

    output:
    tuple val('example'), path('example.train.data.txt'), path('example.train.labels.txt'), emit: train_datasets
    tuple val('example'), path('example.test.data.txt'), path('example.test.labels.txt'), emit: test_datasets

    script:
    """
    make-dataset.py \
        --n-samples  ${params.n_samples} \
        --n-features ${params.n_features} \
        --n-classes  ${params.n_classes}
    """
}
