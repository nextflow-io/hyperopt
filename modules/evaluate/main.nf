nextflow.enable.types = true

process EVALUATE {
    tag "${dataset_name}/${model_type}"

    input:
    record(
        model_type: String,
        model: Path,
        dataset_name: String,
        data: Path,
        meta: Path
    )

    output:
    record(
        model_type: model_type,
        dataset_name: dataset_name,
        score: fromJson(file('score.json')) as Map,
        logs: file('evaluate.log'),
    )

    script:
    """
    evaluate.py \
        --model ${model} \
        --data  ${data} \
        --meta  ${meta} \
        > evaluate.log
    """
}

/*
 * Load data from a JSON file
 */
def fromJson(file: Path) {
    return new groovy.json.JsonSlurper().parse(file)
}
