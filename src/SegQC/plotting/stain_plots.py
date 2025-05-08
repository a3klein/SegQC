import os
import sys
import numpy as np
import polars as pl
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pypalettes import load_cmap
import datashader as ds
import colorcet

seg_qc_path = "/ceph/cephatlas/aklein/SegQC/src"
sys.path.append(seg_qc_path)
from SegQC.Stains import Stains
from SegQC.Transcripts import Transcripts
from SegQC.plotting import table_plots
from SegQC.constants import X_LOC, Y_LOC, Z_LOC, CELL_ID, GENE_NAME, GENE_ID, FOV_COL

# For plotting an image
def plot_img(img, grid=True, flip=False, title:str="", ax=None): 
    if ax == None: 
        _, ax = plt.subplots(figsize=(15, 15))
    if flip: 
        img = np.flipud(img)
    ax.imshow(img,     
              vmin = np.percentile(img, 99)*0.01,
              vmax = np.percentile(img, 99)*1.1,
              cmap = sns.dark_palette("#bfcee3", reverse=False, as_cmap=True)
    )
    if grid:
        ax.grid(visible=True, which='both', axis='both', c='#fff', ls='--', lw=0.5)
    ax.set_title(title)
    
    plt.gca().set_aspect('equal')
    
    return ax

def plot_stains(stains:Stains, reg_name="", plot_report=False, pdfFile=None, cmap_name:str="Klein", 
                figsize=(15, 10)): 
    """
    A function for plotting all of the stains in a Stains object

    attributes: 
        stains - an object of type Stains
        plot_report - whether to plot the stats table from the report 
        pdfFile - if not None, then save the report to this pdf file
    """
    cmap = load_cmap(cmap_name)
    n_channels = stains.get_num_channels()

    ncols = 3
    nrows = int( (n_channels + 1)/ 3) + 1
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)

    for i, img_name in enumerate(stains.get_channels()):
        color = cmap.colors[i]
        img = stains.get_image(img_name)
        ax = fig.add_subplot(gs[int(i / 3), int(i % 3)])
        ax = plot_img(img, grid=False, ax=ax)
        ax.set_title(img_name, color=color, size=20)

    if plot_report: 
        # Getting the report 
        df_report = stains.gen_image_report()
        cols = df_report.columns
        rows = df_report.select('Stain').to_numpy().flatten()
        data = df_report.select(cols).to_numpy()
        rcolors = cmap.colors[:len(rows)]

        def fmt_string(x): 
            return [a if type(a) == str else "{:.3E}".format(a) for a in x]
        data = np.apply_along_axis(fmt_string, axis=1, arr=data)
        
        axb = fig.add_subplot(gs[nrows - 1, :])
        axb = table_plots.plot_table_on_ax(data, rows, cols, title="",rcolors=rcolors, ax=axb)

        if pdfFile: 
            pdfFile.savefig(fig)
        
    return fig

def plot_tz(Tr:Transcripts, reg_name="", pdf_file=None, kde_overlay=False, df_kde=None, ax=None):
    """
    """
    df_tz = Tr.tr_df.to_pandas()
    cvs = ds.Canvas(plot_width=900, plot_height=900)
    agg = cvs.points(df_tz, X_LOC, Y_LOC)
    img = ds.tf.shade(agg, cmap=colorcet.fire, how='log')
    if ax is None: 
        fig, ax = plt.subplots(figsize=(15,15), facecolor='grey')
    ax.imshow(img.to_pil(),  extent=[df_tz[X_LOC].min(), df_tz[X_LOC].max(),
                                     df_tz[Y_LOC].min(), df_tz[Y_LOC].max()])
    if kde_overlay: 
        # sns.kdeplot(df_kde.sample(1000), x='global_x', y='global_y', levels=10, color='#00FF00', alpha=1, linewidths=2, ax=ax)
        sns.kdeplot(df_kde, x=X_LOC, y=Y_LOC, fill=True, alpha=0.65, thresh=0, levels=100, cmap="mako", ax=ax)
    ax.axis('off')
    ax.set_title(f"{reg_name} - Num Transcripts={df_tz.shape[0]}")
    if pdf_file: 
        pdf_file.savefig(fig, bbox_inches='tight')
    else: 
        return ax
    

def plot_tz_report(Tr:Transcripts, reg_name="", pdfFile=None, control_genes:list = None, figsize=(10, 6)): 
    """
    """
    if control_genes is not None: 
        df_kde = Tr.tr_df.filter(pl.col(GENE_NAME).is_in(control_genes)).to_pandas()

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize, constrained_layout=True, facecolor="grey")
    
    plot_tz(Tr, reg_name, ax=axes[0])
    if control_genes is not None: 
        plot_tz(Tr, reg_name=reg_name, kde_overlay=True, df_kde=df_kde, ax=axes[1])

    if pdfFile:
        pdfFile.savefig(fig)
    return fig, axes


def plot_fov(img, bbox, ax=None):
    # using the plot_img function for plotting
    ax = plot_img(img, grid=False, flip=False, title="", ax=ax)
    
    # getting the bounding box
    xmin, ymin, xmax, ymax = bbox    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax