
process SPLIT_TRAIN_TEST {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${file}" }
    tag "${dataset_name}"

    input:
    tuple val(dataset_name), path(meta_file), path(data_file)

    output:
    tuple val(dataset_name), path(meta_file), path('train.txt'), path('test.txt')

    script:
    """
    split-train-test.py --data ${data_file}
    """
}
