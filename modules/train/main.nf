nextflow.enable.types = true

process TRAIN {
    tag "${dataset_name}/${model_type}"

    input:
    record(
        dataset_name: String,
        meta: Path,
        data: Path,
        model_type: String
    )

    output:
    record(
        dataset_name: dataset_name,
        model_type: model_type,
        model: file('model.pkl'),
        logs: file('train.log'),
    )

    script:
    """
    train.py \
        --data       ${data} \
        --meta       ${meta} \
        --scaler     standard \
        --model-type ${model_type} \
        > train.log
    """
}
