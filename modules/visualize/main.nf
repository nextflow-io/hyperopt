
process visualize {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${file}" }

    input:
    tuple val(dataset_name), path(data_file), path(meta_file)

    output:
    tuple val(dataset_name), path('*.png'), emit: plots

    script:
    """

    visualize.py \
        --data    ${data_file} \
        --meta    ${meta_file} \
        --outfile `basename ${data_file} .txt`.png
    """
}
