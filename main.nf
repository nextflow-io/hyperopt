#!/usr/bin/env nextflow 

nextflow.enable.moduleBinaries = true

include { FETCH_DATASET } from './modules/fetch_dataset'
include { SPLIT_TRAIN_TEST } from './modules/split_train_test'
include { VISUALIZE as VISUALIZE_TRAIN } from './modules/visualize'
include { VISUALIZE as VISUALIZE_TEST } from './modules/visualize'
include { TRAIN } from './modules/train'
include { EVALUATE } from './modules/evaluate'


/*
 * Pipeline parameters. They can be overriden on the command line,
 * e.g. `--fetch_datasets some_value`.
 */
params.fetch_datasets = null
params.train_test_splits = null
params.train_models = null
params.pretrained_models = null
params.outdir = 'results'

/* 
 * entry workflow
 */
workflow {
    log.info """\
      H Y P E R O P T   P I P E L I N E
      =================================
      fetch_datasets    : ${params.fetch_datasets}
      train_test_splits : ${params.train_test_splits}
      train_models      : ${params.train_models}
      pretrained_models : ${params.pretrained_models}
      outdir            : ${params.outdir}
    """.stripIndent()

    // fetch and split datasets if specified
    if( params.fetch_datasets != null ) {
        ch_dataset_names = channel.fromList(params.fetch_datasets.tokenize(','))
        ch_datasets = FETCH_DATASET(ch_dataset_names)
        ch_train_test_splits = SPLIT_TRAIN_TEST(ch_datasets)
    }

    // otherwise load custom train/test splits
    else if( params.train_test_splits != null ) {
        ch_datasets = channel.empty()
        ch_train_test_splits = channel.of(file(params.train_test_splits))
            .flatMap { json -> json.splitJson() }
            .map { r ->
                tuple(r.dataset_name, file(r.meta), file(r.data_train), file(r.data_test))
            }
    }

    else {
        error "Either `--fetch_datasets` or `--train_test_splits` must be provided (run with `-profile test` to use default test data)"
    }

    // separate training and test data
    ch_train_datasets = ch_train_test_splits.map { dataset_name, meta, data_train, data_test ->
        tuple(dataset_name, meta, data_train)
    }
    ch_test_datasets = ch_train_test_splits.map { dataset_name, meta, data_train, data_test ->
        tuple(dataset_name, meta, data_test)
    }

    // visualize train/test datasets
    VISUALIZE_TRAIN(ch_train_datasets)
    VISUALIZE_TEST(ch_test_datasets)

    // print warning if both training and pre-trained model are enabled
    if( params.train_models != null && params.pretrained_models != null ) {
        log.warn 'Pre-trained model(s) were provided but training is also enabled -- pre-trained models will be ignored'
    }

    // train new models if specified
    if( params.train_models != null ) {
        model_types = params.train_models.tokenize(',')
        (ch_models, ch_train_logs) = TRAIN(ch_train_datasets, model_types)
    }

    // otherwise load pretrained models if specified
    else if( params.pretrained_models != null ) {
        ch_models = channel.of(file(params.pretrained_models))
            .flatMap { json -> json.splitJson() }
            .map { r ->
                tuple(r.dataset_name, r.model_type, file(r.model))
            }
    }

    else {
        error "Either `--train_models` or `--pretrained_models` must be provided (run with `-profile test` to use default test data)"
    }

    // evaluate each model against test dataset
    ch_evaluate_inputs = ch_models.combine(ch_test_datasets, by: 0)
    (ch_scores, ch_test_logs) = EVALUATE(ch_evaluate_inputs)

    // report the best model for each dataset based on evaluation score
    ch_scores
        .max { it -> fromJson(it[2]).value }
        .subscribe { dataset_name, model_type, score_file ->
            def score = fromJson(score_file)
            printf "The best model for dataset '${dataset_name}' was '${model_type}' (${score.name} = %.3f)\n", score.value
        }
}


def fromJson(file) {
    return new groovy.json.JsonSlurper().parse(file)
}
