
process split_train_test {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${file}" }
    tag "${dataset_name}"

    input:
    tuple val(dataset_name), path(data_file), path(labels_file)

    output:
    tuple val(dataset_name), path('train.data.txt'), path('train.labels.txt'), emit: train_datasets
    tuple val(dataset_name), path('test.data.txt'), path('test.labels.txt'), emit: test_datasets

    script:
    """
    split-train-test.py \
        --data ${data_file} \
        --labels ${labels_file}
    """
}
