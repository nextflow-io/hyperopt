
process FETCH_DATASET {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${file}" }
    tag "${dataset_name}"

    input:
    val(dataset_name)

    output:
    tuple val(dataset_name), path('meta.json'), path('data.txt')

    script:
    """
    fetch-dataset.py --name ${dataset_name}
    """
}
