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

include { train } from './modules/train'
include { predict } from './modules/predict'
include { visualize } from './modules/visualize'
include { make_dataset } from './modules/make_dataset'


log.info """
    M L - E X A M P L E   P I P E L I N E
    =====================================
    make_dataset    : ${params.make_dataset}
    n_samples       : ${params.n_samples}
    n_features      : ${params.n_features}
    n_classes       : ${params.n_classes}

    visualize       : ${params.visualize}

    train           : ${params.train}
    train_data      : ${params.train_data}
    train_labels    : ${params.train_labels}
    train_models    : ${params.train_models}

    predict         : ${params.predict}
    predict_model   : ${params.predict_model}
    predict_data    : ${params.predict_data}
    predict_labels  : ${params.predict_labels}

    outdir          : ${params.outdir}
    """


/* 
 * main script flow
 */
workflow {
    // create synthetic data if specified
    if ( params.make_dataset == true ) {
        (ch_train_datasets, ch_predict_datasets) = make_dataset()
    }

    // otherwise load input files
    else {
        ch_train_data = Channel.fromFilePairs(params.train_data, size: 1, flat: true)
        ch_train_labels = Channel.fromFilePairs(params.train_labels, size: 1, flat: true)
        ch_train_datasets = train_data.join(train_labels)

        ch_predict_data = Channel.fromFilePairs(params.predict_data, size: 1, flat: true)
        ch_predict_labels = Channel.fromFilePairs(params.predict_labels, size: 1, flat: true)
        ch_predict_datasets = predict_data.join(predict_labels)
    }

    if ( params.visualize == true ) {
        visualize(ch_train_datasets.concat(ch_predict_datasets))
    }

    // print warning if both training and pre-trained model are enabled
    if ( params.train == true && params.predict_model != null ) {
        log.warn 'Training is enabled but pre-trained model is also provided, pre-trained model will be ignored'
    }

    // perform training if specified
    if ( params.train == true ) {
        ch_models = train(ch_train_datasets, params.train_models)
    }

    // otherwise load trained model if specified
    else if ( params.predict_model != null ) {
        ch_models = Channel.fromFilePairs(params.predict_model, size: 1, flat: true)
            | map { [it[0], 'pretrained', it[1]]}
    }

    // perform inference if specified
    if ( params.predict == true ) {
        ch_predict_inputs = ch_models.combine(ch_predict_datasets, by: 0)
        predict(ch_predict_inputs)
    }
}


/* 
 * completion handler
 */
workflow.onComplete {
	log.info ( workflow.success ? '\nDone!' : '\nOops .. something went wrong' )
}
