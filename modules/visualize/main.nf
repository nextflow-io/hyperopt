
process VISUALIZE {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${file}" }

    input:
    tuple val(dataset_name), path(meta_file), path(data_file)

    output:
    tuple val(dataset_name), path('*.png')

    script:
    """
    visualize.py \
        --data    ${data_file} \
        --meta    ${meta_file} \
        --outfile `basename ${data_file} .txt`.png
    """
}
