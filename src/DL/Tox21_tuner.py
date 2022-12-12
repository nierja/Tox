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
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, roc_auc_score, auc, accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, fbeta_score, recall_score

spacer = '------------------------------------------------------------------------'
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--n_layers", default=3, type=int, help="Number of hidden layers")
parser.add_argument("--cv", default=10, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--NN_type", default="DNN", type=str, help="Type of a NN architecture")
parser.add_argument("--fp", default="mordred", type=str, help="Fingerprint to use")
parser.add_argument("--pca", default=0, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args: argparse.Namespace) -> list:
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

    # Optionaly, merge the train and validation datasets
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

    def model_builder(hp, metrics=METRICS, output_bias=OUTPUT_BIAS):
        # build the desired model in a sequential manner

        hp_units1 = hp.Int('units1', min_value=256, max_value=512, step=256)
        hp_units2 = hp.Int('units2', min_value=256, max_value=512, step=256)

        model = keras.Sequential([
            keras.layers.Dense(
                units=hp_units1, activation='relu',
                input_shape=(train_features.shape[-1],)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(
                units=hp_units2, activation='relu',
                ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid',
                            bias_initializer=output_bias),
        ])

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

        return model

    # get the class weights
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # Instantiate the hyperband tuner and perform the tuning
    log_dir = "logs/hp/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    tuner = kt.Hyperband(model_builder,
                        objective=kt.Objective("auc", direction="max"),
                        max_epochs=10,
                        factor=3,
                        directory=log_dir,
                        project_name='intro_to_kt'
                        )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='auc', 
        verbose=1,
        patience=20,
        mode='max',
        restore_best_weights=True)
        
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

    # calculate the optimal number of epochs
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # perform the crossvalidation of the best model
    kfold = KFold(n_splits=args.cv, shuffle=True)
    auc_per_fold = []; loss_per_fold = []

    fold_no = 1
    for train, test in kfold.split(train_features, train_labels):
        history = model.fit(
            train_features[train], 
            train_labels[train],
            epochs=best_epoch,
            class_weight=class_weight, 
            callbacks=[stop_early]
        )
      
        # Generate generalization metrics
        scores = model.evaluate(train_features[test], train_labels[test], verbose=0)
        auc = scores[9]
        print(f'Score for fold {fold_no}: "auc" = {auc}')
        print(spacer)
        auc_per_fold.append(scores[9])
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no += 1

    print(f'{args.cv}-fold CV: "auc_CV" = {np.array(auc_per_fold).mean()}+-{np.array(auc_per_fold).std()}')
    print(f'{args.cv}-fold CV: "loss_CV" = {np.array(loss_per_fold).mean()}+-{np.array(loss_per_fold).std()}')

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

    # finally, evaluate the ensamble performance
    eval_result = ensemble_model.evaluate(test_features, test_labels)
    print(spacer)
    print(ensemble_model.metrics_names)
    print("METRICS:", eval_result)
    print(spacer)
    return 0

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)