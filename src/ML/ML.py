#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from tabulate import tabulate
import grid_search_parameters
import plotting

import sklearn.cluster
import sklearn.compose
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.impute import SimpleImputer 
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.random_projection import SparseRandomProjection

import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
gpus = tf.config.list_physical_devices('GPU'); logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs available")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_classes", default=2, type=int, help="Number of target classes")
parser.add_argument("--cv", default=3, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--roc", default=False, type=bool, help="Plot the ROC_AUCs")
parser.add_argument("--vis", default=None, type=str, help="Visualisation type [Isomap|NCA|SRP|tSVD|TSNE]")
parser.add_argument("--pca", default=False, type=bool, help="Plot the PCAs")
parser.add_argument("--pca_comps", default=20, type=int, help="dimensionality of space the dataset is reduced to using pca")
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--model", default="lr", type=str, help="Model to use")
parser.add_argument("--scaler", default="StandardScaler", type=str, help="defines scaler to preprocess data")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args: argparse.Namespace) -> list:
    # We are training a model.
    np.random.seed(args.seed)
    positive_PCA_features, positive_vis_features = [], []
    negative_PCA_features, negative_vis_features = [], []
    results, y_true, y_predicted_proba, fprs, tprs = [], [], [], [], []

    fpdict_keys = ['maccs']
    # fpdict_keys = ['rdkit_descr', 'ecfp0','ecfp2', 'ecfp4', 'fcfp2', 'fcfp4']
    # fpdict_keys = ['dist_2D' ,'dist_3D', 'adjac', 'inv_dist_2D', 'inv_dist_3D', 'Laplacian']
    # fpdict_keys = [ 'rdkit_descr', 'ecfp0','ecfp2', 'ecfp4', 'fcfp2', 'fcfp4', 'CMat_full', 'CMat_400', 'dist_2D' ,
    #               'dist_3D', 'adjac', 'Laplacian', ]
    # fpdict_keys = ['ecfp0','ecfp2', 'ecfp4', 'ecfp6', 'fcfp2', 'fcfp4', 'fcfp6',
    #               'maccs', 'hashap', 'hashtt', 'avalon', 'rdk5', 'rdk6', 'rdk7',
    #               'CMat_full', 'CMat_400', 'CMat_600', 'eigenvals', 'dist_2D' ,
    #               'dist_3D', 'balaban_2D', 'balaban_3D', 'adjac', 'Laplacian', ]

    for fp_name in fpdict_keys:
        print(fp_name, args.model, args.target)
        # load the dataset
        df_train = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{fp_name}.data")
        df_test = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{fp_name}_test.data")
        df_eval = pd.read_csv(f"../../data/Tox21_descriptors/{args.target}/{args.target}_{fp_name}_eval.data")

        #  convert it into numpy arrays
        data_train, target_train = df_train.iloc[:, 0:-2].to_numpy(), df_train.iloc[:, -1].to_numpy()
        data_test, target_test = df_test.iloc[:, 0:-2].to_numpy(), df_test.iloc[:, -1].to_numpy()
        final_evaluation_data, final_evaluation_target = df_eval.iloc[:, 0:-2].to_numpy(), df_eval.iloc[:, -1].to_numpy()
        
        # merge df_train and df_test for grid search
        data = np.concatenate((data_train, data_test), axis=0)
        target = np.concatenate((target_train, target_test), axis=0)

        # perfoms the PCA transformation to R^{args.pca_comps} space
        if args.pca:
            transformer = IncrementalPCA(n_components=args.pca_comps)
            data = sparse.csr_matrix(data)
            data = transformer.fit_transform(data)
            if args.pca_comps == 2:
                positive_PCA_features.append(data[target[:] == 1])
                negative_PCA_features.append(data[target[:] == 0])

        # train a model on the given dataset and store it in 'model'.
        if args.model in ["most_frequent", "stratified"]:
            model = sklearn.dummy.DummyClassifier(strategy=args.model)
        elif args.model == "gbt":
            model = sklearn.ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=20, verbose=1)
        else:
            if args.model == "lr":
                model = [
                    ("lr_cv", sklearn.linear_model.LogisticRegressionCV(max_iter=100)),
                ]
            elif args.model == "svm":
                model = [
                    ("svm", sklearn.svm.SVC(max_iter=100, probability=True, verbose=1, kernel="linear")),
                ]
            elif args.model == "adalr":
                model = [
                    ("ada_lr_cv", sklearn.ensemble.AdaBoostClassifier(sklearn.linear_model.LogisticRegression(C=1), n_estimators=5)),
                ]
            elif args.model == "baglr":
                model = [
                    ("bag_lr_cv", sklearn.ensemble.BaggingClassifier(sklearn.linear_model.LogisticRegression(C=1), n_estimators=5)),
                ]
            elif args.model == "badlr":
              model = [("lr", sklearn.linear_model.LogisticRegression())]
            elif args.model == "mlp":
                model = [
                    ("mlp", sklearn.neural_network.MLPClassifier(tol=0, learning_rate_init=0.01, max_iter=20, hidden_layer_sizes=(100), activation="relu", solver="adam", verbose=1)),
                ]

            # int_columns = []
            float_columns = list(range(0,data.shape[1]))
            # print(float_columns)
            if args.scaler == "StandardScaler": 
                model = sklearn.pipeline.Pipeline([
                    ("preprocess", sklearn.compose.ColumnTransformer([
                        ("scaler", sklearn.preprocessing.StandardScaler(), float_columns),
                    ]))
                ] + model)
            if args.scaler == "MinMaxScaler": 
                model = sklearn.pipeline.Pipeline([
                    ("preprocess", sklearn.compose.ColumnTransformer([
                        ("scaler", sklearn.preprocessing.MinMaxScaler(), float_columns),
                    ]))
                ] + model)
            if args.scaler == "MaxAbsScaler": 
                model = sklearn.pipeline.Pipeline([
                    ("preprocess", sklearn.compose.ColumnTransformer([
                        ("scaler", sklearn.preprocessing.MaxAbsScaler(), float_columns),
                    ]))
                ] + model)

        # perform grid search with CV and find best parameters
        parameters = grid_search_parameters.get_grid_search_parameters(args.model)
        grid_search = GridSearchCV(model, parameters, n_jobs=-1, )
        print(data, data.shape)
        print(data[0][-1])
        grid_search.fit(data, target)
        print(f"\nBest hyperparameters:\n{grid_search.best_params_}\n")

        # perform prediction on the final eval dataset using the best model
        final_evaluation_predictions = grid_search.best_estimator_.predict(final_evaluation_data)
        final_evaluation_proba = grid_search.best_estimator_.predict_proba(final_evaluation_data)

        # compute the desired metrics
        accuracy = accuracy_score(final_evaluation_target, final_evaluation_predictions)
        balanced_accuracy = balanced_accuracy_score(final_evaluation_target, final_evaluation_predictions)
        print(accuracy, balanced_accuracy)
        break

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(args.n_classes):
            fpr[i], tpr[i], _ = roc_curve(test_target.ravel(), test_proba[:, i].ravel())
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area - micro-average NOT OK
        fpr["micro"], tpr["micro"], _ = roc_curve(test_target.ravel(), (test_proba[:,1]).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # store data for future plotting and console output
        results.append((fp_name, accuracy, balanced_accuracy, roc_auc["micro"])); 
        y_true.append(test_target); y_predicted_proba.append(test_proba)
        fprs.append(fpr["micro"]); tprs.append(tpr["micro"])

        # perfoms selected projection of training data to R^2 space
        if args.vis != None:
            # load the training data anew, since they were distorted with PCA
            df = pd.read_csv(f"Tox21_data/{args.target}/{args.target}_{fp_name}.data")
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
            data, target = df.iloc[:, 0:-2].to_numpy(), df.iloc[:, -1].to_numpy()
            imputer = SimpleImputer(missing_values=np.nan,strategy = "mean") 
            data = imputer.fit_transform(data)

            if args.vis == "Isomap": data = Isomap(n_components=args.n_classes).fit_transform(data)
            if args.vis == "NCA": data = NeighborhoodComponentsAnalysis(n_components=args.n_classes, init="pca",).fit_transform(data, target)
            if args.vis == "SRP": data = SparseRandomProjection(n_components=args.n_classes,).fit_transform(data)
            if args.vis == "TSNE": data = TSNE(n_components=args.n_classes, learning_rate='auto', init='random').fit_transform(data)
            if args.vis == "tSVD": data = TruncatedSVD(n_components=args.n_classes).fit_transform(data)
            positive_vis_features.append(data[target[:] == 1])
            negative_vis_features.append(data[target[:] == 0])

    if args.roc: plotting.plot_ROCs(fprs, tprs, fpdict_keys, nrows=2, ncols=3, model=args.model, target=args.target)
    if args.vis != None: plotting.plot_DimReds(args.vis, positive_vis_features, negative_vis_features, fpdict_keys, nrows=2, ncols=3, model=args.model, target=args.target)
    if args.pca and args.pca_comps == 2: plotting.plot_DimReds("PCA", positive_PCA_features, negative_PCA_features, fpdict_keys, nrows=2, ncols=3, model=args.model, target=args.target)

    return results


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    results = main(args)
    with open(f"../../results/logs/ML_output_{args.target}.txt", "a") as output_file:
        table = tabulate(results, headers=["fp_name", "acc", "balanced_acc", "roc"], floatfmt=(None, '.4f', '.2f',))
        output_file.write(f"\n{args.model} - {args.target} \nTest size = {args.test_size}\n" + table)
    print(table)