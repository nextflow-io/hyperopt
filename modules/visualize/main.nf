nextflow.enable.types = true

process VISUALIZE {
    tag data.name

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
        data: data,
        plot: file('*.png'),
    )

    script:
    """
    visualize.py \
        --data    ${data} \
        --meta    ${meta} \
        --outfile `basename ${data} .txt`.png
    """
}
