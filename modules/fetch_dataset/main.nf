nextflow.enable.types = true

process FETCH_DATASET {
    tag dataset_name

    input:
    dataset_name: String

    output:
    record(
        dataset_name: dataset_name,
        meta: file('meta.json'),
        data: file('data.txt'),
    )

    script:
    """
    fetch-dataset.py --name ${dataset_name}
    """
}
