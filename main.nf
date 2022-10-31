#!/usr/bin/env nextflow 

/*
 * Copyright (c) 2022, Seqera Labs.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This Source Code Form is "Incompatible With Secondary Licenses", as
 * defined by the Mozilla Public License, v. 2.0.
 *
 */
import groovy.json.JsonSlurper

include { fetch_dataset } from './modules/fetch_dataset'
include { split_train_test } from './modules/split_train_test'
include { visualize } from './modules/visualize'
include { train } from './modules/train'
include { predict } from './modules/predict'


log.info """
    M L - E X A M P L E   P I P E L I N E
    =====================================
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
    """


/* 
 * main script flow
 */
workflow {
    // fetch dataset if specified
    if ( params.fetch_dataset == true ) {
        ch_datasets = fetch_dataset(params.dataset_name)

        (ch_train_datasets, ch_predict_datasets) = split_train_test(ch_datasets)
    }

    // otherwise load input files
    else {
        ch_train_data = Channel.fromFilePairs(params.train_data, size: 1, flat: true)
        ch_train_meta = Channel.fromFilePairs(params.train_meta, size: 1, flat: true)
        ch_train_datasets = ch_train_data.join(ch_train_meta)

        ch_predict_data = Channel.fromFilePairs(params.predict_data, size: 1, flat: true)
        ch_predict_meta = Channel.fromFilePairs(params.predict_meta, size: 1, flat: true)
        ch_predict_datasets = ch_predict_data.join(ch_predict_meta)
    }

    // visualize train/test sets
    if ( params.visualize == true ) {
        visualize(ch_train_datasets.concat(ch_predict_datasets))
    }

    // print warning if both training and pre-trained model are enabled
    if ( params.train == true && params.predict_models != null ) {
        log.warn 'Training is enabled but pre-trained model(s) are also provided, pre-trained models will be ignored'
    }

    // perform training if specified
    if ( params.train == true ) {
        (ch_models, ch_train_logs) = train(ch_train_datasets, params.train_models)
    }

    // otherwise load trained model if specified
    else if ( params.predict_models != null ) {
        ch_models = Channel.fromFilePairs(params.predict_models, size: 1, flat: true)
            | map { [it[0], 'pretrained', it[1]] }
    }

    // perform inference if specified
    if ( params.predict == true ) {
        ch_predict_inputs = ch_models.combine(ch_predict_datasets, by: 0)
        (ch_scores, ch_predict_logs) = predict(ch_predict_inputs)

        // select the best model based on inference score
        ch_scores
            | max {
                new JsonSlurper().parse(it[2])['value']
            }
            | subscribe { dataset_name, model_type, score_file ->
                def score = new JsonSlurper().parse(score_file)
                println "The best model for ${dataset_name} was ${model_type}, with ${score['name']} = ${score['value']}"
            }
    }
}


/* 
 * completion handler
 */
workflow.onComplete {
	log.info ( workflow.success ? '\nDone!' : '\nOops .. something went wrong' )
}
