
process predict {
    publishDir params.outdir, mode: 'copy'
    tag "${dataset_name}/${model_type}"

    input:
    tuple val(dataset_name), val(model_type), path(model_file), path(data_file), path(labels_file)

    output:
    tuple val(dataset_name), val(model_type), path("${dataset_name}.predict.${model_type}.log"), emit: logs

    script:
    """
    predict.py \
        --model  ${model_file} \
        --data   ${data_file} \
        --labels ${labels_file} \
        > ${dataset_name}.predict.${model_type}.log
    """
}
