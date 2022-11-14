#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
import grid_search_parameters

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
from sklearn.metrics import make_scorer, roc_auc_score, auc, accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, fbeta_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
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
parser.add_argument("--pca_comps", default=50, type=int, help="dimensionality of space the dataset is reduced to using pca")
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

        # define metrics to be tracked in the grid search
        scoring = {
            "AUC": "roc_auc",
            "acc": make_scorer(accuracy_score),
            "balanced_acc": make_scorer(balanced_accuracy_score),
            "f1_score": make_scorer(f1_score),
        }

        # perform grid search with CV and find best parameters
        parameters = grid_search_parameters.get_grid_search_parameters(args.model)
        grid_search = GridSearchCV(
            model, parameters, cv=args.cv, n_jobs=-1, scoring=scoring, refit="AUC", return_train_score=False,
        )
        grid_search.fit(data, target)
        results = grid_search.cv_results_
        best_params = grid_search.best_params_
        print(results)
        print(f"\nBest hyperparameters:\n{grid_search.best_params_}\n")

        # get crossvalidated metrics for the best model
        scores = cross_validate(
            grid_search.best_estimator_, data, target, cv=args.cv, scoring=scoring,
        )
        # THESE are ok and can be logged
        # print(scores)

        # perform prediction on the final eval dataset using the best model
        final_evaluation_predictions = grid_search.best_estimator_.predict(final_evaluation_data)
        final_evaluation_proba = grid_search.best_estimator_.predict_proba(final_evaluation_data)

        # compute desired metrics for the prediction on the final evaluation set using the best model
        accuracy = accuracy_score(final_evaluation_target, final_evaluation_predictions)
        balanced_accuracy = balanced_accuracy_score(final_evaluation_target, final_evaluation_predictions)
        f1 = f1_score(final_evaluation_target, final_evaluation_predictions)
        roc_auc = roc_auc_score(final_evaluation_target, final_evaluation_proba[:,1])

        # print(final_evaluation_target, final_evaluation_predictions)
        print(accuracy, balanced_accuracy, f1, roc_auc)
        
        # print(f1_score(final_evaluation_target, final_evaluation_predictions, labels=[0,1], average='micro'))
        # print(f1_score(final_evaluation_target, final_evaluation_predictions, labels=[0,1], average='macro'))
        # print(f1_score(final_evaluation_target, final_evaluation_predictions, labels=[0,1], average='weighted'))        

        # print(precision_recall_fscore_support(final_evaluation_target, final_evaluation_predictions, average='macro'), labels=[0,1])
        # print(precision_recall_fscore_support(final_evaluation_target, final_evaluation_predictions, average='micro'), labels=[0,1])
        # print(precision_recall_fscore_support(final_evaluation_target, final_evaluation_predictions, average='weighted'), labels=[0,1])
        # print(precision_recall_fscore_support(final_evaluation_target, final_evaluation_predictions, average='binary'), labels=[0,1])

        # print(recall_score(final_evaluation_target, final_evaluation_predictions))
        # print(f1_score(final_evaluation_target, final_evaluation_predictions))
        # print(fbeta_score(final_evaluation_target, final_evaluation_predictions, beta=0.5))
        # print(fbeta_score(final_evaluation_target, final_evaluation_predictions, beta=1))
        # print(fbeta_score(final_evaluation_target, final_evaluation_predictions, beta=2))
        # print(precision_recall_fscore_support(final_evaluation_target, final_evaluation_predictions, beta=0.5))

        # log data into a csv file
        file_path = f'../../results/logs/ML_{args.target}.csv'
        if not os.path.isfile(file_path): 
            # create a csv header if the file doesn't exist
            with open(file_path, 'w') as f:
                print(
                    "dataset;model;model_info;fp;pca;"\
                    "best_val_auc;crossval_auc;crossval_auc_std;"\
                    "best_balanced_acc;crossval_balanced_acc;crossval_balanced_acc_std;"\
                    "best_acc;crossval_acc;crossval_acc_std;"\
                    "best_F1;crossval_F1;crossval_F1_std;"\
                    "best_params", file=f
                )

        with open(file_path, 'a') as f:
            print(
            f"Tox21;{args.model};-;{fp_name};{args.pca};"\
            f"{roc_auc};{scores['test_AUC'].mean()};{scores['test_AUC'].std()};"\
            f"{balanced_accuracy};{scores['test_balanced_acc'].mean()};{scores['test_balanced_acc'].std()};"\
            f"{accuracy};{scores['test_acc'].mean()};{scores['test_acc'].std()};"\
            f"{f1};{scores['test_f1_score'].mean()};{scores['test_f1_score'].std()};"\
            f"{best_params}", file=f
        )
    
        # print to stdout as well
        print(f"fp={fp_name}, pca={args.pca}, best_validation_roc_auc={roc_auc}, best_params={best_params}")
        print(f"{args.cv}-fold crossvalidation auc = {scores['test_AUC'].mean()} +- {scores['test_AUC'].std()}")
        print(
            f"Tox21;{args.model};-;{fp_name};{args.pca};"\
            f"{roc_auc};{scores['test_AUC'].mean()};{scores['test_AUC'].std()};"\
            f"{balanced_accuracy};{scores['test_balanced_acc'].mean()};{scores['test_balanced_acc'].std()};"\
            f"{accuracy};{scores['test_acc'].mean()};{scores['test_acc'].std()};"\
            f"{f1};{scores['test_f1_score'].mean()};{scores['test_f1_score'].std()};"\
            f"{best_params}"
        )
    return 0


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)