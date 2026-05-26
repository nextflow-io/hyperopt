#!/usr/bin/env nextflow 

nextflow.enable.moduleBinaries = true
nextflow.enable.types = true

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
params {
    fetch_datasets: String?
    train_test_splits: Path?
    train_models: String?
    pretrained_models: Path?
}

/* 
 * entry workflow
 */
workflow {
    main:
    log.info """\
      H Y P E R O P T   P I P E L I N E
      =================================
      fetch_datasets    : ${params.fetch_datasets}
      train_test_splits : ${params.train_test_splits}
      train_models      : ${params.train_models}
      pretrained_models : ${params.pretrained_models}
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
        ch_train_test_splits = channel.of(params.train_test_splits)
            .flatMap { json -> json.splitJson() as List<Map> }
            .map { r ->
                record(
                    dataset_name: r.dataset_name as String,
                    meta: file(r.meta, checkIfExists: true),
                    data_train: file(r.data_train, checkIfExists: true),
                    data_test: file(r.data_test, checkIfExists: true)
                )
            }
    }

    else {
        error "Either `--fetch_datasets` or `--train_test_splits` must be provided (run with `-profile test` to use default test data)"
    }

    // separate training and test data
    ch_train_datasets = ch_train_test_splits.map { r ->
        record(dataset_name: r.dataset_name, meta: r.meta, data: r.data_train)
    }
    ch_test_datasets = ch_train_test_splits.map { r ->
        record(dataset_name: r.dataset_name, meta: r.meta, data: r.data_test)
    }

    // visualize train/test datasets
    ch_train_plots = VISUALIZE_TRAIN(ch_train_datasets).map { r ->
        record(dataset_name: r.dataset_name, plot_train: r.plot)
    }
    ch_test_plots = VISUALIZE_TEST(ch_test_datasets).map { r ->
        record(dataset_name: r.dataset_name, plot_test: r.plot)
    }

    // combine train/test data with train/test plots
    ch_splits = ch_train_test_splits
        .join(ch_train_plots, by: 'dataset_name')
        .join(ch_test_plots, by: 'dataset_name')

    // print warning if both training and pre-trained model are enabled
    if( params.train_models != null && params.pretrained_models != null ) {
        log.warn 'Pre-trained model(s) were provided but training is also enabled -- pre-trained models will be ignored'
    }

    // train new models if specified
    if( params.train_models != null ) {
        model_types = params.train_models.tokenize(',')
        ch_train_inputs = ch_train_datasets.flatMap { r ->
            model_types.collect { model_type ->
                r + record(model_type: model_type)
            }
        }
        ch_models = TRAIN(ch_train_inputs)
    }

    // otherwise load pretrained models if specified
    else if( params.pretrained_models != null ) {
        ch_models = channel.of(params.pretrained_models)
            .flatMap { json -> json.splitJson() as List<Map> }
            .map { r ->
                record(
                    dataset_name: r.dataset_name as String,
                    model_type: r.model_type as String,
                    model: file(r.model, checkIfExists: true)
                )
            }
    }

    else {
        error "Either `--train_models` or `--pretrained_models` must be provided (run with `-profile test` to use default test data)"
    }

    // evaluate each model against test dataset
    ch_evaluate_inputs = ch_models.join(ch_test_datasets, by: 'dataset_name')
    ch_evals = EVALUATE(ch_evaluate_inputs).map { r ->
        record(
            model_type: r.model_type,
            dataset_name: r.dataset_name,
            score_name: r.score.name,
            score: r.score.value,
            logs: r.logs,
        )
    }

    // report the best model for each dataset based on evaluation score
    ch_evals
        .map { r -> tuple(r.dataset_name, r) }
        .groupBy()
        .subscribe { dataset_name, evals ->
            def best = evals.max { r -> r.score }
            printf "The best model for dataset '${dataset_name}' was '${best.model_type}' (${best.score_name} = %.3f)\n", best.score
        }

    publish:
    datasets = ch_datasets
    splits = ch_splits
    trained_models = params.train_models != null ? ch_models : channel.empty()
    evals = ch_evals
}

/*
 * Workflow outputs
 */
output {
    datasets: Channel<Dataset> {
        path { r -> "datasets/${r.dataset_name}/" }
        index {
            path 'datasets/index.json'
        }
    }

    splits: Channel<TrainTestSplit> {
        path { r -> "splits/${r.dataset_name}/" }
        index {
            path 'splits/index.json'
        }
    }

    trained_models: Channel<Model> {
        path { r -> "trained_models/${r.dataset_name}-${r.model_type}/" }
        index {
            path 'trained_models/index.json'
        }
    }

    evals: Channel<Eval> {
        path { r -> "evals/${r.dataset_name}-${r.model_type}/" }
        index {
            path 'evals/index.json'
        }
    }
}

/*
 * Types
 */
record Dataset {
    dataset_name: String
    meta: Path
    data: Path
}

record TrainTestSplit {
    dataset_name: String
    meta: Path
    data_train: Path
    plot_train: Path
    data_test: Path
    plot_test: Path
}

record Model {
    model_type: String
    model: Path
    dataset_name: String
}

record Eval {
    model_type: String
    dataset_name: String
    score_name: String
    score: Float
}
