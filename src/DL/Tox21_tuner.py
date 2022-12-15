#!/usr/bin/env python3
"""
Original tutorial
    https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/keras_tuner.ipynb

"""

import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
import keras_tuner as kt

import argparse
import numpy as np
import pandas as pd
import os
import sys
import datetime

from scipy import sparse
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, roc_auc_score, auc, accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, fbeta_score, recall_score

spacer = '------------------------------------------------------------------------'
metrics_dict = {
    "auc" : [],
    "prc" : [],
    "acc" : [],
    "f1"  : [],
    "ba"  : [],
    "tp"  : [],
    "fp"  : [],
    "tn"  : [],
    "fn"  : [],
}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--n_layers", default=3, type=int, help="Number of hidden layers")
parser.add_argument("--cv", default=3, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--target", default="SR-MMP", type=str, help="Target toxocity type")
parser.add_argument("--NN_type", default="DNN", type=str, help="Type of a NN architecture")
parser.add_argument("--fp", default="maccs", type=str, help="Fingerprint to use")
parser.add_argument("--weighted", default=False, type=bool, help="Set class weights")
parser.add_argument("--pca", default=0, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def log_scores(dictionary, metrics_names, scores) -> None:
    # logs desired metrics into a dictionary
    for name, value in zip(metrics_names, scores):
        if name == "tp" : tp = value
        if name == "fp" : fp = value
        if name == "tn" : tn = value
        if name == "fn" : fn = value

    tpr = tp / ( tp + fn )
    tnr = tn / ( tn + fp )
    f1 = tp / ( tp + 0.5 * ( fp + fn ) )
    ba = ( tpr + tnr ) / 2

    auc_idx = metrics_names.index( "auc" )
    acc_idx = metrics_names.index( "accuracy" )
    prc_idx = metrics_names.index( "prc" ) 

    dictionary["auc"].append( scores[ auc_idx ] )
    dictionary["acc"].append( scores[ acc_idx ] )
    dictionary["prc"].append( scores[ prc_idx ] )
    dictionary["f1"].append( f1 )
    dictionary["ba"].append( ba )
    dictionary["tp"].append( tp )
    dictionary["fp"].append( fp )
    dictionary["tn"].append( tn )
    dictionary["fn"].append( fn )


def main(args: argparse.Namespace) -> int:
    """

    Args:
        args (argparse.Namespace): A dictionary of parameters to modify the ML workflow

    Returns:
        int: 0 is retuned upon a succesful run
    """
    # We are training a model.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df_train = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{args.fp}.data")
    df_test = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{args.fp}_test.data")
    df_eval = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{args.fp}_eval.data")

    #  convert dataframes into numpy arrays
    train_features, train_labels = df_train.iloc[:, 0:-2].to_numpy(), df_train.iloc[:, -1].to_numpy()
    val_features, val_labels = df_test.iloc[:, 0:-2].to_numpy(), df_test.iloc[:, -1].to_numpy()
    test_features, test_labels = df_eval.iloc[:, 0:-2].to_numpy(), df_eval.iloc[:, -1].to_numpy()

    # perfoms the PCA transformation to R^{args.pca} space
    if args.pca:
        print(train_features.shape, val_features.shape, test_features.shape)
        data = np.concatenate((train_features, val_features, test_features), axis=0)
        transformer = IncrementalPCA(batch_size=2048, n_components=args.pca)
        data = sparse.csr_matrix(data)
        data = transformer.fit_transform(data)

        train_features = data[0:train_features.shape[-2],:]
        val_features = data[train_features.shape[-2]:train_features.shape[-2]+val_features.shape[-2],:]
        test_features = data[train_features.shape[-2]+val_features.shape[-2]:,:]
        print(train_features.shape, val_features.shape, test_features.shape)

    # Optionaly, merge the train and validation datasets
    print(train_labels)
    train_labels = np.concatenate((train_labels, val_labels), axis=0)
    train_features = np.concatenate((train_features, val_features), axis=0)

    # count the number of positive and negative molecules
    neg, pos = np.bincount(train_labels.astype(int))
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Normalize the dataset by scaling and clipping
    scaler = StandardScaler()

    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    # print the dataset shape
    print('Training labels shape:', train_labels.shape)
    print('Validation labels shape:', val_labels.shape)
    print('Test labels shape:', test_labels.shape)

    print('Training features shape:', train_features.shape)
    print('Validation features shape:', val_features.shape)
    print('Test features shape:', test_features.shape)

    # calculate the optimal initial bias
    initial_bias = np.log([pos/neg])
    print(f"initial bias={initial_bias}")
    OUTPUT_BIAS = tf.keras.initializers.Constant(initial_bias)

    # define the metrics used
    threshold = 0.5
    METRICS = [
        keras.metrics.TruePositives(name='tp', thresholds=[threshold]),
        keras.metrics.FalsePositives(name='fp', thresholds=[threshold]),
        keras.metrics.TrueNegatives(name='tn', thresholds=[threshold]),
        keras.metrics.FalseNegatives(name='fn', thresholds=[threshold]), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'), 
    ]


    """
    def model_builder(hp, metrics=METRICS, output_bias=OUTPUT_BIAS):
        # build the desired model in a sequential manner

        model = keras.Sequential()
        model.add(keras.layers.Flatten())
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, args.n_layers)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=256, max_value=1024, step=256),
                    activation=hp.Choice("activation", ["relu"]),
                )
            )
        if hp.Boolean("dropout"):
            model.add(keras.layers.Dropout(rate=0.25))

        model.add(keras.layers.Dense(1, activation='sigmoid',
                        bias_initializer=output_bias))

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-3, sampling="log")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )
        return model
    """

    def model_builder(hp, metrics=METRICS, output_bias=OUTPUT_BIAS):
        # DUMMY FUNCTION, for logging only
        # DUMMY FUNCTION, for logging only
        # DUMMY FUNCTION, for logging only        
        # build the desired model in a sequential manner

        model = keras.Sequential()
        model.add(keras.layers.Flatten())
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 1)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=512, max_value=512, step=512),
                    activation=hp.Choice("activation", ["relu"]),
                )
            )
        
        model.add(keras.layers.Dense(1, activation='sigmoid',
                        bias_initializer=output_bias))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )
        return model


    # get the class weights
    if args.weighted:
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
    else:
        class_weight = {0: 1, 1: 1}

    # Instantiate the hyperband tuner and perform the tuning
    log_dir = "logs/hp/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    # DUMMY TUNER, for logging only
    # DUMMY TUNER, for logging only
    # DUMMY TUNER, for logging only

    tuner = kt.Hyperband(model_builder,
                        objective=kt.Objective("auc", direction="max"),
                        max_epochs=2,
                        factor=3,
                        directory=log_dir,
                        project_name='intro_to_kt',
                        )

    """
    tuner = kt.Hyperband(model_builder,
                        objective=kt.Objective("auc", direction="max"),
                        max_epochs=10,
                        factor=3,
                        directory=log_dir,
                        project_name='intro_to_kt',
                        )
    """

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='auc', 
        verbose=1,
        patience=20,
        mode='max',
        restore_best_weights=True
    )
        
    tuner.search(train_features, train_labels, epochs=50, validation_split=0.2, callbacks=[stop_early,tensorboard_callback])

    # get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(best_hps)
    print(tuner.search_space_summary())
    print(tuner.results_summary()
    )

    # Train the model, find the optimal number of epochs to train 
    # the model with the hyperparameters obtained from the search.
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_features, train_labels, epochs=50, validation_split=0.2, class_weight=class_weight, )
    # TODO: Add logging
    # print(model.history)
    # print(model.history.history['val_auc'][-1])
    # sys.exit()

    # calculate the optimal number of epochs
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    """
        Perform k-fold crossvalidation on the best model
        
    """
    kfold = KFold(n_splits=args.cv, shuffle=True)
    CV_metrics = metrics_dict.copy()    # creates a deep copy of metrics_dict

    fold_no = 1
    for train, test in kfold.split(train_features, train_labels):
        
        model = tuner.hypermodel.build(best_hps)

        history = model.fit(
            train_features[train], 
            train_labels[train],
            epochs=best_epoch,
            class_weight=class_weight, 
            callbacks=[stop_early]
        )
      
        # Generate generalization metrics
        scores = model.evaluate(train_features[test], train_labels[test], verbose=0)

        log_scores( CV_metrics, model.metrics_names, scores )

        print(spacer)
        print(f'{args.cv}-fold CV -- fold no. {fold_no} RESULTS :\n')
        for key in CV_metrics.keys():
            print(key, ': ', (CV_metrics[key])[-1] )
        print(spacer)  
        print(model.metrics_names, scores, CV_metrics)

        # Increase fold number
        fold_no += 1

    print(f'{args.cv}-fold CV: "auc_CV" = {np.array(CV_metrics["auc"]).mean()}+-{np.array(CV_metrics["auc"]).std()}')

    # recreate the best model several times and traind the individual
    # instances for the best number of epochs
    hypermodel1 = tuner.hypermodel.build(best_hps)
    hypermodel2 = tuner.hypermodel.build(best_hps)
    hypermodel3 = tuner.hypermodel.build(best_hps)
    # hypermodel4 = tuner.hypermodel.build(best_hps)
    # hypermodel5 = tuner.hypermodel.build(best_hps)
    # hypermodel6 = tuner.hypermodel.build(best_hps)
    # hypermodel7 = tuner.hypermodel.build(best_hps)
    # hypermodel8 = tuner.hypermodel.build(best_hps)
    # hypermodel9 = tuner.hypermodel.build(best_hps)
    # hypermodel10 = tuner.hypermodel.build(best_hps)
    # hypermodel11 = tuner.hypermodel.build(best_hps)

    hypermodels = [
        hypermodel1,
        hypermodel2,
        hypermodel3, ]
    #     hypermodel4,
    #     hypermodel5,
    #     hypermodel6,
    #     hypermodel7,
    #     hypermodel8,
    #     hypermodel9,
    #     hypermodel10,
    #     hypermodel11,
    # ]

    # Retrain the models with optimal number of epochs
    for hypermodel in hypermodels:
        hypermodel.fit(train_features, train_labels, epochs=best_epoch, validation_split=0.2, class_weight=class_weight, callbacks=[stop_early])

    plot_model(hypermodel1, show_shapes=True, to_file='hypermodel1_plot.png')
    hypermodel1.save("single_model")

    # evaluate the hypermodels on the test data individually
    for hypermodel in hypermodels:
        eval_result = hypermodel.evaluate(test_features, test_labels)
        print(spacer)
        print("METRICS:", eval_result)

    # create an ensamble model from the hypermodles
    models = hypermodels
    model_input = tf.keras.Input(shape=(train_features.shape[-1],))
    model_outputs = [model(model_input) for model in models]
    ensemble_output = tf.keras.layers.Average()(model_outputs)
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    # compile and plot the ensamble model
    ensemble_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
    plot_model(ensemble_model, show_shapes=True, to_file='ensemble_model_plot.png')
    ensemble_model.save("ensemble_model")

    # finally, evaluate the ensamble performance
    eval_result = ensemble_model.evaluate(test_features, test_labels)
    print(spacer)
    print('ENSEMBLE MODEL RESULTS:\n')
    for name, value in zip(ensemble_model.metrics_names, eval_result):
            print(name, ': ', value)
    print(spacer)

    FP = []
    FN = []
    test_predictions = ensemble_model.predict(test_features)
    for idx, prediction in enumerate ( test_predictions ):
        if ( ( prediction > 0.5 ) and ( test_labels [ idx ] == 0 ) ):
            FP.append ( prediction[0] )
        if ( ( prediction < 0.5 ) and ( test_labels [ idx ] == 1 ) ):
            FN.append ( prediction[0] )

    print(f'fn : {FN}')
    print(f'fp : {FP}')

    # log data into a csv file
    file_path = f'../../results/logs/DL_{args.target}.csv'
    if not os.path.isfile(file_path): 
        # create a csv header if the file doesn't exist
        ...

    with open(file_path, 'a') as f:
        # log parameters
        ...
    
    # print to STDOUT
    ...

    return 0

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)