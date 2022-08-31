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
nextflow.enable.dsl = 2


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


process make_dataset {
    publishDir params.outdir, mode: 'copy'

    output:
    tuple val('example'), path('example.train.data.txt'), path('example.train.labels.txt'), emit: train_datasets
    tuple val('example'), path('example.test.data.txt'), path('example.test.labels.txt'), emit: test_datasets

    script:
    """
    make-dataset.py \
        --n-samples  ${params.n_samples} \
        --n-features ${params.n_features} \
        --n-classes  ${params.n_classes}
    """
}


process visualize {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(dataset_name), path(data_file), path(labels_file)

    output:
    tuple val(dataset_name), path('*.png'), emit: plots

    script:
    """
    visualize.py \
        --data    ${data_file} \
        --labels  ${labels_file} \
        --outfile `basename ${data_file} .txt`.png
    """
}


process train {
    publishDir params.outdir, mode: 'copy'
    tag { "${dataset_name}/${model_type}" }

    input:
    tuple val(dataset_name), path(data_file), path(labels_file)
    each model_type

    output:
    tuple val(dataset_name), val(model_type), path("${dataset_name}.pkl"), emit: models

    script:
    """
    train.py \
        --data       ${data_file} \
        --labels     ${labels_file} \
        --scaler     standard \
        --model-type ${model_type} \
        --model-name ${dataset_name}.pkl
    """
}


process predict {
    publishDir params.outdir, mode: 'copy'
    tag { "${dataset_name}/${model_type}" }

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


/* 
 * completion handler
 */
workflow.onComplete {
	log.info ( workflow.success ? '\nDone!' : '\nOops .. something went wrong' )
}
