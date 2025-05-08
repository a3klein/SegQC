
# Imports
import os
import sys

import numpy as np
import polars as pl

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm 
import seaborn as sns

from adjustText import adjust_text
from pypalettes import load_cmap

from SegQC.Transcripts import Transcripts
from SegQC.constants import X_LOC, Y_LOC, Z_LOC, CELL_ID, GENE_NAME, GENE_ID, FOV_COL

def plot_voxel_dist(Tr:Transcripts, pdfFile=None, ax=None): 
    """
    attributes: 
        pdfFile(str) : if not None, save the figure to this pdf
        ax(Matplotlib.Axes) : if not None, plot the figure on this ax
    """
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    sns.histplot(Tr.voxel_dist, ax=ax)
    ax.set_title("Distribution of Transcript Density in voxels of 10 pixels^2")
    ax.set_xlabel("Number of Transcripts")
    return ax


def plot_tr_balance(Tr:Transcripts, scale=True, plot_PCA=False, pdfFile=None, ax=None): 
    """
    attributes: 
        scale(bool) : whether to scale the data (zscore it)
        plot_PCA(bool)
        pdfFile(str) : if not None, save the figure to this pdf
        ax(Matplotlib.Axes) : if not None, plot the figure on this ax
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde

    def makeColors(vals): 
        colors = np.zeros((len(vals), 3))
        norm = Normalize(vmin=vals.min(), vmax = vals.max())
        colors = [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]
        return colors
    
    if scale: 
        ss = StandardScaler()
        np_mean_pos = ss.fit_transform(Tr.tr_mean_pos[X_LOC, Y_LOC].to_numpy()).T
    else: 
        np_mean_pos = Tr.tr_mean_pos[X_LOC, Y_LOC].to_numpy().T
    
    denseObj = gaussian_kde(np_mean_pos)
    cc = makeColors(denseObj.evaluate(np_mean_pos))

    if ax is None: 
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.scatter(np_mean_pos[0], np_mean_pos[1], s=1, color=cc)

    if plot_PCA:
        pca = PCA(n_components=2).fit(np_mean_pos)
        for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
            ax.plot([0, comp[0]], [0, comp[1]], label=f'C{i}, var:{var:.2f}',
                    color=f'C{i+2}', alpha=0.5)
        plt.legend()

    ax.set_title("Mean transcript position")
    ax.set_xlabel(X_LOC)
    ax.set_ylabel(Y_LOC)

    # calculating the density of each point and filtering the least dense points 
    density_scores = denseObj.evaluate(np_mean_pos)
    thr = np.percentile(density_scores, 1)
    to_label = np.where(density_scores < thr)[0]

    # getting the plot positions and gene names to plot
    x_pos, y_pos = np_mean_pos[:, to_label]
    text = Tr.tr_mean_pos.with_row_index().filter(pl.col("index").is_in(to_label)).select(GENE_NAME).to_numpy().flatten()

    red_c = load_cmap("Abbott").colors[0]
    texts = []
    for x, y, t in zip(x_pos, y_pos, text): 
        texts.append(ax.text(x, y, t, ha="center", va="center", c=red_c))

    adjust_text(texts, ax=ax)
    return ax

def plot_tr_cdf(Tr, regress = True, n_genes = 3, pdfFile=None, ax=None): 
    """
    attributes:
        regress(bool) : whether to plot the regression line
        n_genes(int) : the number of genes to plot
        pdfFile(str) : if not None, save the figure to this pdf
        ax(Matplotlib.Axes) : if not None, plot the figure on this ax
    """
    from sklearn.linear_model import LinearRegression

    # for the linear regression 
    Y = np.log(Tr.tr_gene_vc["count"].to_numpy())
    X = np.arange(len(Y)).reshape(-1, 1)
    model = LinearRegression().fit(X, Y)
    R2 = model.score(X, Y)
    pred = model.predict(X)

    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.scatter(np.arange(Tr.tr_gene_vc.shape[0]), Tr.tr_gene_vc['count'].to_numpy(), color='blue', s=1)
    ax.plot(np.arange(Tr.tr_gene_vc.shape[0]), np.exp(pred), color='red')

    # Getting text
    text = list(Tr.tr_gene_vc.head(n_genes).select(GENE_NAME).to_numpy().flatten())
    text.extend(list(Tr.tr_gene_vc.tail(n_genes).select(GENE_NAME).to_numpy().flatten()))
    # Getting the positions for the text
    x = [a[0]+1 for a in X[:n_genes]]
    y = [np.exp(a) for a in Y[:n_genes]]
    x.extend([a[0] for a in X[-n_genes:]])
    y.extend([np.exp(a) for a in Y[-n_genes:]])

    red_c = load_cmap("Abbott").colors[0]
    texts = []
    for xl, yl, t in zip(x, y, text): 
        texts.append(ax.text(xl, yl, t, ha='center', va='center', c=red_c))

    ax.set_title("Transcript Sorted Counts (R2: %.3f)" % R2)
    ax.set_yscale("log")
    ax.set_ylabel("Number of transcripts (log scale)")
    ax.set_xlabel("Gene ID")

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color=red_c, lw=0.5),
                expand=(1.5, 1.5),  expand_axes=True, force_statis=0.5, ax=ax,
                )
    return ax


def plot_base_pipeline(Tr:Transcripts, pdfFile=None, figsize=(15, 4)): 
        """
        attributes: 
            pdfFile(str) : if not None, save the figure to this pdf
        """
        # Make a plot: 
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=figsize)

        plot_tr_cdf(Tr, regress=True, n_genes=3, ax=axes[0])
        plot_tr_balance(Tr, scale=True, plot_PCA=False, ax=axes[1])
        plot_voxel_dist(Tr, ax=axes[2])
        
        #pdfFile.close()
        if pdfFile:
            plt.savefig(pdfFile, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            return fig, axes