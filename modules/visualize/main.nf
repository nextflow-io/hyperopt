
process visualize {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(dataset_name), path(data_file), path(labels_file)

    output:
    tuple val(dataset_name), path('*.png'), emit: plots

    script:
    """
    visualize.py \
        --data    ${data_file} \
        --labels  ${labels_file} \
        --outfile `basename ${data_file} .txt`.png
    """
}
