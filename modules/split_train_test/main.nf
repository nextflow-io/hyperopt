nextflow.enable.types = true

process SPLIT_TRAIN_TEST {
    tag dataset_name

    input:
    record(
        dataset_name: String,
        meta: Path,
        data: Path
    )

    output:
    record(
        dataset_name: dataset_name,
        meta: meta,
        data_train: file('train.txt'),
        data_test: file('test.txt'),
    )

    script:
    """
    split-train-test.py --data ${data}
    """
}
