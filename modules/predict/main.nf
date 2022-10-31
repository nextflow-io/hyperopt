
process predict {
    publishDir params.outdir, mode: 'copy', saveAs: { file -> "${dataset_name}.${model_type}.${file}" }
    tag "${dataset_name}/${model_type}"

    input:
    tuple val(dataset_name), val(model_type), path(model_file), path(data_file), path(meta_file)

    output:
    tuple val(dataset_name), val(model_type), path('score.json'), emit: scores
    tuple val(dataset_name), val(model_type), stdout, emit: logs

    script:
    """
    predict.py \
        --model ${model_file} \
        --data  ${data_file} \
        --meta  ${meta_file}
    """
}
