#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_ROCs(fprs: list, tprs: list, keys: list, nrows: int, ncols: int, model: str, target: str) -> None:
    # create and save a multiplot of ROCs for all fingerprints
    fig, axs = plt.subplots(nrows, ncols, figsize=(2*ncols,2*nrows), constrained_layout=True); lw = 2
    for i in range(nrows):
        for j in range(ncols):
            axs[i, j].plot(
                # fprs[(ncols*i+j)%(nrows*ncols)],
                # tprs[(ncols*i+j)%(nrows*ncols)],
                fprs[(ncols*i+j)%len(fprs)],
                tprs[(ncols*i+j)%len(fprs)],
                color="crimson",
                lw=lw,
                label="ROC_AUC = %0.2f" % auc(fprs[(ncols*i+j)%len(fprs)], tprs[(ncols*i+j)%len(fprs)]),
            )
            axs[i, j].set_title(keys[(ncols*i+j)%len(fprs)])
            axs[i, j].legend(loc="lower right", prop={'size': 6})
            axs[i, j].plot([0, 1], [0, 1], color="blue", lw=lw, linestyle="--")
    for ax in axs.flat:
        ax.set(xlabel='FPR', ylabel='TPR')
        ax.set(adjustable='box', aspect='equal')
        ax.label_outer()
    #plt.show()
    plt.savefig(f'./Plots/ROC_plot_{model}_{target}.png', dpi=1200)

def plot_DimReds(reduction_type: str, positive: list, negative: list, keys: list, nrows: int, ncols: int, model: str, target: str) -> None:
    # create and save a multiplot of PCA graphs for all the fingerprints
    fig, axs = plt.subplots(nrows, ncols, figsize=(2*ncols,2*nrows), constrained_layout=True); lw = 2
    for i in range(nrows):
        for j in range(ncols):
            axs[i, j].scatter(
                negative[(ncols*i+j)%len(positive)][:,0],
                negative[(ncols*i+j)%len(positive)][:,1],
                c="b",
                s=1,
            )
            axs[i, j].scatter(
                positive[(ncols*i+j)%len(positive)][:,0],
                positive[(ncols*i+j)%len(positive)][:,1],
                c="r",
                s=1,
            )
            axs[i, j].set_title(keys[(ncols*i+j)%len(positive)])
    for ax in axs.flat:
        ax.set(xlabel=f'{reduction_type} feature 1', ylabel=f'{reduction_type} feature 2')
        #ax.set(adjustable='box', aspect='equal')
        ax.label_outer()
    plt.savefig(f'./Plots/{reduction_type}_plot_{model}_{target}.png', dpi=1200)
