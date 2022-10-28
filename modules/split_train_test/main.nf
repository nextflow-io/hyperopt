
process split_train_test {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${file}" }
    tag "${dataset_name}"

    input:
    tuple val(dataset_name), path(data_file), path(meta_file)

    output:
    tuple val(dataset_name), path('train.txt'), path(meta_file), emit: train_datasets
    tuple val(dataset_name), path('test.txt'), path(meta_file), emit: test_datasets

    script:
    """
    split-train-test.py --data ${data_file}
    """
}
