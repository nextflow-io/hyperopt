
process fetch_dataset {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${file}" }
    tag "${dataset_name}"

    input:
    val(dataset_name)

    output:
    tuple val(dataset_name), path('data.txt'), path('meta.json'), emit: datasets

    script:
    """
    fetch-dataset.py --name ${dataset_name}
    """
}
