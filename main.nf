#!/usr/bin/env nextflow 

nextflow.enable.moduleBinaries = true

include { fetch_dataset } from './modules/fetch_dataset'
include { split_train_test } from './modules/split_train_test'
include { visualize } from './modules/visualize'
include { train } from './modules/train'
include { predict } from './modules/predict'


/* 
 * entry workflow
 */
workflow {
    log.info """\
      M L - H Y P E R O P T   P I P E L I N E
      =======================================
      fetch_dataset   : ${params.fetch_dataset}
      dataset_name    : ${params.dataset_name}

      visualize       : ${params.visualize}

      train           : ${params.train}
      train_data      : ${params.train_data}
      train_meta      : ${params.train_meta}
      train_models    : ${params.train_models}

      predict         : ${params.predict}
      predict_models  : ${params.predict_models}
      predict_data    : ${params.predict_data}
      predict_meta    : ${params.predict_meta}

      outdir          : ${params.outdir}
    """.stripIndent()

    // fetch dataset if specified
    if ( params.fetch_dataset ) {
        ch_datasets = fetch_dataset(params.dataset_name)

        (ch_train_datasets, ch_predict_datasets) = split_train_test(ch_datasets)
    }

    // otherwise load input files
    else {
        ch_train_data = channel.fromFilePairs(params.train_data, size: 1, flat: true)
        ch_train_meta = channel.fromFilePairs(params.train_meta, size: 1, flat: true)
        ch_train_datasets = ch_train_data.join(ch_train_meta)

        ch_predict_data = channel.fromFilePairs(params.predict_data, size: 1, flat: true)
        ch_predict_meta = channel.fromFilePairs(params.predict_meta, size: 1, flat: true)
        ch_predict_datasets = ch_predict_data.join(ch_predict_meta)
    }

    // visualize train/test sets
    if ( params.visualize ) {
        visualize(ch_train_datasets.concat(ch_predict_datasets))
    }

    // print warning if both training and pre-trained model are enabled
    if ( params.train && params.predict_models != null ) {
        log.warn 'Training is enabled but pre-trained model(s) are also provided, pre-trained models will be ignored'
    }

    // perform training if specified
    if ( params.train ) {
        (ch_models, ch_train_logs) = train(ch_train_datasets, params.train_models)
    }

    // otherwise load trained model if specified
    else if ( params.predict_models != null ) {
        ch_models = channel.fromFilePairs(params.predict_models, size: 1, flat: true)
            .map { [it[0], 'pretrained', it[1]] }
    }

    // perform inference if specified
    if ( params.predict ) {
        ch_predict_inputs = ch_models.combine(ch_predict_datasets, by: 0)
        (ch_scores, ch_predict_logs) = predict(ch_predict_inputs)

        // select the best model based on inference score
        ch_scores
            .max {
                new groovy.json.JsonSlurper().parse(it[2])['value']
            }
            .subscribe { dataset_name, model_type, score_file ->
                def score = new groovy.json.JsonSlurper().parse(score_file)
                println "The best model for \'${dataset_name}\' was \'${model_type}\', with ${score.name} = ${String.format('%.3f', score.value)}"
            }
    }
}
