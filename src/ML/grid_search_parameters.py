#!/usr/bin/env python3

def get_grid_search_parameters(model: str) -> dict:
    # given a model name, returns apropriate set of hyperparameters
    # to be used for calling sklearn.GridSearchCV()
    if model == "gbt":
        parameters = { 
            "max_depth" : [2, 4, 6],
            "n_estimators" : [50, 100, 150],
            "subsample" : [1.0],
            "n_iter_no_change" : [None],
        }
    elif model == "lr":
        parameters = { 
            "lr_cv__Cs" : [1, 10],
            "lr_cv__cv" : [5, 10],
            "lr_cv__max_iter" : [50, 100, 200],
            "lr_cv__n_jobs" : [-1],
        }
    elif model == "svm":
        parameters = { 
            "svm__C" : [0.01, 0.1, 1, 10],
            "svm__max_iter" : [-1, 100],
        }
    elif model == "adalr":
        parameters = { 
            "ada_lr_cv__learning_rate" : [0.01, 0.1, 1, 10],
            "ada_lr_cv__n_estimators" : [5, 10, 20, 50],
        }
    elif model == "baglr":
        parameters = { 
            "bag_lr_cv__n_estimators" : [5, 10, 20, 50],
            "bag_lr_cv__max_samples" : [0.5, 1],
            "bag_lr_cv__max_features" : [0.5, 1],
            "bag_lr_cv__n_jobs" : [-1],
        }
    elif model == "badlr":
        parameters = { 
            "lr__C" : [5, 10, 20, 50],
            "lr__max_iter" : [50, 100],
            "lr__solver" : ["newton-cg", "lbfgs", "saga"],
            "lr__n_jobs" : [-1],
        }
    elif model == "mlp":
        parameters = { 
            "mlp__alpha" : [0.0001],
            "mlp__learning_rate_init" : [0.01, 0.1, 1],
            "mlp__hidden_layer_sizes" : [50, 100, 150],
            "mlp__max_iter" : [25, 50, 100],
        }
    return parameters