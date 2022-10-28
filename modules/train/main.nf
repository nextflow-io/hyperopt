
process train {
    publishDir params.outdir, mode: 'copy'
    tag "${dataset_name}/${model_type}"

    input:
    tuple val(dataset_name), path(data_file), path(meta_file)
    each model_type

    output:
    tuple val(dataset_name), val(model_type), path("${dataset_name}.pkl"), emit: models

    script:
    """
    train.py \
        --data       ${data_file} \
        --meta       ${meta_file} \
        --scaler     standard \
        --model-type ${model_type} \
        --model-name ${dataset_name}.pkl
    """
}
