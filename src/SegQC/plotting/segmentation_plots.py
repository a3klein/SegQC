
import os 
import sys
import json 
import pathlib

import polars as pl
import geopandas as gpd
import shapely
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from adjustText import adjust_text
from pypalettes import load_cmap

from SegQC.constants import CELL_X, CELL_Y, CELL_FOV, CELL_VOLUME, CELL_ID, GENE_NAME, GENE_ID, GEOMETRY_COL



def plot_feature_distribution(df:pl.DataFrame, feature:str, feature_alias:str=None, min_val=None, max_val=None,
                              num_bins:int=100, pdfFile=None, ax=None): 
    """
    attributes: 
        feature(str) : the feature to plot
        feature_alias(str) : the name to plot for the feature
        min_val(float) : the minimum value for the feature
        max_val(float) : the maximum value for the feature
        pdfFile(str) : if not None, save the figure to this pdf
        ax(Matplotlib.Axes) : if not None, plot the figure on this ax
    """
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    sns.histplot(df[feature].to_numpy(), bins=num_bins, ax=ax)
    if min_val: 
        ax.axvline(min_val, color='red', linestyle='--')
    if max_val: 
        ax.axvline(max_val, color='red', linestyle='--')

    ax.set_title(f"{feature} Distribution")
    if feature_alias: 
        ax.set_title(f"{feature_alias} Distribution")

    ax.set_xlabel(feature)
    return ax

def plot_filt_cells(df_feature:pl.DataFrame, title="", pdfFile=None, ax=None): 
    """
    attributes: 
        pdfFile(str) : if not None, save the figure to this pdf
        ax(Matplotlib.Axes) : if not None, plot the figure on this ax
    """
    filt_palette = {True : "#507B58", False : "#540202"}
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    # Plotting the filtered cells
    sns.scatterplot(ax=ax, data=df_feature, x=CELL_X, y=CELL_Y, hue='pass_qc', s=1, palette=filt_palette)
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax

# xmin ymin xmax ymax - order for bbox
def plot_cell_seg_fov(gdf, mmpt, bbox, scale_factor=1, ax=None):
    # getting the bounding box 
    xmin, ymin, xmax, ymax = bbox
    container = shapely.geometry.Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    # applying the affine transformation + scaling to the size of the images
    shapes = gdf[GEOMETRY_COL].apply(lambda x: shapely.affinity.affine_transform(x, [*mmpt[:2, :2].flatten(), *mmpt[:2, 2].flatten()]))
    shapes = shapes.scale(xfact = 1/scale_factor, yfact = 1/scale_factor, zfact=1/scale_factor, origin=(0,0,0))
    # intersecting with the bounding box
    fov_shapes = shapes.loc[shapes.intersects(container)]
    # plotting the shapes on the axis
    fov_shapes.plot(facecolor="b", edgecolor='r', linewidth=2, alpha=0.2, ax=ax, aspect=1)
    return ax


# def plot_cell_density_distribution(df:pl.DataFrame, pdfFile=None, ax=None): 
#     """
#     attributes:
#         pdfFile(str) : if not None, save the figure to this pdf
#         ax(Matplotlib.Axes) : if not None, plot the figure on this ax
#     """

#     # Datashader option 
#     import datashader as ds
#     import colorcet

#     density = df_feature.select([CELL_X, CELL_Y]).to_pandas()
#     cvs = ds.Canvas(plot_width=900, plot_height=900)
#     agg = cvs.points(density, CELL_X, CELL_Y)
#     img = ds.tf.shade(agg, cmap=colorcet.dimgray, how="eq_hist")

#     fig, ax = plt.subplots(figsize=(10,10), facecolor='white')
#     ax.imshow(img.to_pil(),  extent=[density[CELL_X].min(), density[CELL_X].max(),
#                                     density[CELL_Y].min(), density[CELL_Y].max()])

#     ax.axis('off')
#     ax.set_title(f"Num Cells={density.shape[0]}")
#     plt.show()

#     ### 2D heatmap option 
#     density = df_feature.select([CELL_X, CELL_Y]).to_numpy()
#     heatmap, xedges, yedges = np.histogram2d(density[:, 0], density[:, 1], bins=200)
#     fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
#     ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
#             origin='lower', cmap='viridis')
#     ax.set_title("Cell Density Distribution")
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
#     plt.show()
    
#     return ax