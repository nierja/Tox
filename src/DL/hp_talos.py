#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

import sklearn.preprocessing
import sklearn.metrics
from scipy import sparse
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score


import os
import tensorflow as tf
from tensorflow import keras
import talos

gpus = tf.config.list_physical_devices('GPU'); logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs available")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--n_layers", default=3, type=int, help="Number of hidden layers")
parser.add_argument("--cv", default=3, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--NN_type", default="DNN", type=str, help="Type of a NN architecture")
parser.add_argument("--fp", default="maccs", type=str, help="Fingerprint to use")
parser.add_argument("--pca", default=20, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--param_fraction", default=0.001, type=lambda x:int(x) if x.isdigit() else float(x), help="Fraction of all paremeter configurations to try out")


def main(args: argparse.Namespace) -> list:
    # We are training a model.
    np.random.seed(args.seed)

    print(args.fp, args.NN_type, args.target)
    # load the dataset
    df_train = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{args.fp}.data")
    df_test = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{args.fp}_test.data")
    df_eval = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{args.fp}_eval.data")

    #  convert it into numpy arrays
    data_train, target_train = df_train.iloc[:, 0:-2].to_numpy(), df_train.iloc[:, -1].to_numpy()
    data_test, target_test = df_test.iloc[:, 0:-2].to_numpy(), df_test.iloc[:, -1].to_numpy()
    final_evaluation_data, final_evaluation_target = df_eval.iloc[:, 0:-2].to_numpy(), df_eval.iloc[:, -1].to_numpy()
        
    # merge df_train and df_test for grid search
    data = np.concatenate((data_train, data_test), axis=0)
    target = np.concatenate((target_train, target_test), axis=0)

    # perfoms the PCA transformation to R^2 space
    if args.pca:
        transformer = IncrementalPCA(n_components=args.pca)
        data = sparse.csr_matrix(data)
        data = transformer.fit_transform(data)
    # for CNN, transform the data into 3D tensor
    if args.NN_type == "CNN":
        data.reshape(data.shape[0],data.shape[1],1)
        print(data.shape)

    # scale the data to have zero mean and unit variance
    scaler = sklearn.preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    final_evaluation_data = scaler.fit_transform(final_evaluation_data)

    # splitting dataset into a train set and a test set.
    train_data, validation_data, train_target, validation_target = train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # create parameter dict
    p = {
        'activation': ['relu'],
        'batch_size': [64, 128],
        'epochs': [10,50,100], 
    }
    
    # fill the dictionary up
    if args.NN_type == "DNN":
        for i in range(1, args.n_layers+1):
            p[f'hidden_layer_{i}'] = [50, 100, data.shape[1]//4, data.shape[1]//2]
            p[f'dropout_layer_{i}'] = [0.0, 0.5]
    if args.NN_type == "CNN":
        for i in range(1, args.n_layers+1):
            p[f'conv_layer_{i}_filter'] = [2*i, 4*i, 8*i]
            p[f'conv_layer_{i}_kernel'] = [3, 5]
        p['conv_hidden_layer'] = [data.shape[1]//8, data.shape[1]//4, data.shape[1]//2]
        p['conv_dropout'] = [0, 0.5]

    def tox_model(train_data, train_target, test_data, test_target, params):
        model = keras.Sequential()

        # define the model architecture
        if args.NN_type == "DNN":
            model.add(keras.layers.Dense(data.shape[1], input_shape=(data.shape[1],), activation=params['activation']))
            for i in range(1, args.n_layers+1):
                model.add(keras.layers.Dense(params[f'hidden_layer_{i}'], activation='relu'))
                model.add(keras.layers.Dropout(params[f'dropout_layer_{i}']))

        if args.NN_type == "CNN":
            model.add(keras.layers.Conv1D(
                params[f'conv_layer_1_filter'], 
                params[f'conv_layer_1_kernel'], 
                input_shape=(data.shape[1],1), 
                activation='relu',
                )
            )
            model.add(keras.layers.MaxPooling1D())
            for i in range(2, args.n_layers+1):
                model.add(keras.layers.Conv1D(
                    params[f'conv_layer_{i}_filter'], 
                    params[f'conv_layer_{i}_kernel'], 
                    activation='relu'),
                )
                model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(params['conv_hidden_layer'], activation=tf.nn.relu))
            model.add(keras.layers.Dropout(params['conv_dropout']))

        # add one neuron to the output layer to perform the binary classification
        model.add(keras.layers.Dense(args.n_classes-1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[
                        'AUC', 
                        'accuracy', 
                        talos.utils.metrics.f1score,
                        # sklearn.metrics.balanced_accuracy_score,
                        # tfma.metrics.F1Score, 
                        # tfma.metrics.BalancedAccuracy,
                        # tfma.metrics.BinaryAccuracy,
                    ])

        out = model.fit(x=train_data, 
                y=train_target,
                validation_data=[test_data, test_target],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0,
                callbacks=[talos.utils.early_stopper(params['epochs'])],
            )

        return out, model

    # perform the grid search, with early stopping
    scan_object = talos.Scan(
        x=train_data, 
        y=train_target, 
        params=p, 
        model=tox_model, 
        experiment_name=f'dnn_hparams_logs', 
        fraction_limit=args.param_fraction,
        print_params=False,
    )
    # perform analysis of the results
    analyze_object = talos.Analyze(scan_object)

    # get the best model and its parameters
    best_model = analyze_object.data[analyze_object.data.val_auc == analyze_object.data.val_auc.max()]
    best_params = p.copy()
    for key in best_params:
        if "kernel" in key:
            best_params[key] = (int(best_model.iloc[0][key]))
        else:
            best_params[key] = best_model.iloc[0][key]
    print(best_params)
    
    # perform k-fold crossvalidation
    # as in https://stackoverflow.com/questions/66695848/kfold-cross-validation-in-tensorflow
    auc_scores = []; accuracy_scores = []; balanced_accuracy_scores = []; F1_scores = []
    for kfold, (train, test) in enumerate(KFold(n_splits=args.cv, 
                                    shuffle=True).split(data, target)):
        # clear the session 
        tf.keras.backend.clear_session()

        # get the model
        _, seq_model = tox_model(train_data, train_target, validation_data, validation_target, best_params)

        # run the model 
        seq_model.fit(
            data[train], 
            target[train],
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            validation_data=(data[test], target[test]),
            verbose=0,
        )

        # get the F1score, auc and acc metrics
        print(f"*** history val_auc: {seq_model.history.history['val_auc'][-1]}")
        auc_scores.append(seq_model.history.history['val_auc'][-1])
        accuracy_scores.append(seq_model.history.history['val_accuracy'][-1])
        F1_scores.append(seq_model.history.history['val_f1score'][-1])
        
        # calculate pridictions on the validation set to get the validation balanced accuracy
        predictions = seq_model.predict(data[test])
        test_predictions = (predictions > 0.5).astype("int32")
        _ = np.ones((predictions.shape))
        test_proba = np.column_stack((_, predictions))
        test_proba[:, 0] -= test_proba[:, 1]
        balanced_accuracy = balanced_accuracy_score(target[test], test_predictions)
        print(balanced_accuracy)
        print("\n\n")


        balanced_accuracy_scores.append(balanced_accuracy)
        # seq_model.save_weights(f'wg_{args.cv}.txt')

    print(auc_scores)
    # log data into a csv file
    file_path = f'./Results/talos_hp_results_{args.target}.csv'
    if not os.path.isfile(file_path): 
        # create a csv header if the file doesn't exist
        with open(file_path, 'w') as f:
            print(
                "NN;NN_layers;fp;pca;\
                best_val_auc;crossval_auc;crossval_auc_std;\
                crossval_balanced_acc;crossval_balanced_acc_std;\
                crossval_acc;crossval_acc_std;\
                crossval_F1;crossval_F1_std;\
                best_params", file=f
            )

    with open(file_path, 'a') as f:
        print(
        f"{args.NN_type};{args.n_layers};{args.fp};{args.pca};\
        {analyze_object.data.val_auc.max()};{np.array(auc_scores).mean()};{np.array(auc_scores).std()};\
        {np.array(balanced_accuracy_scores).mean()};{np.array(balanced_accuracy_scores).std()};\
        {np.array(accuracy_scores).mean()};{np.array(accuracy_scores).std()};\
        {np.array(F1_scores).mean()};{np.array(F1_scores).std()};\
        {best_params}", file=f
    )
    
    # print to stdout as well
    print(f"fp={args.fp}, pca={args.pca}, val_auc={analyze_object.data.val_auc.max()}, best_params={best_params}")
    print(f"{args.cv}-fold crossvalidation auc = {np.array(auc_scores).mean()} +- {np.array(auc_scores).std()}")
    print(
        f"{args.NN_type};{args.n_layers};{args.fp};{args.pca};"\
        f"{analyze_object.data.val_auc.max()};{np.array(auc_scores).mean()};{np.array(auc_scores).std()};"\
        f"{np.array(balanced_accuracy_scores).mean()};{np.array(balanced_accuracy_scores).std()};"\
        f"{np.array(accuracy_scores).mean()};{np.array(accuracy_scores).std()};"\
        f"{np.array(F1_scores).mean()};{np.array(F1_scores).std()};"\
        f"{best_params}"\
    )
    return 0

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)