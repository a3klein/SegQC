# system
import os
import sys
import json, gzip
from pathlib import Path


# data 
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import shapely
import anndata as ad
import scipy as sp

import matplotlib.pyplot as plt

from SegQC.plotting.segmentation_plots import plot_feature_distribution, plot_filt_cells
from SegQC.constants import DEFAULT_PRESET, PROSEG_PRESET
from SegQC.constants import CELL_X, CELL_Y, CELL_FOV, CELL_VOLUME, CELL_ID, GEOMETRY_COL
                 
class SegMask(): 
    """
    A class to handle the output of a segmentation algorithm 
    Not specific to a particular algorithm, however assumes some commonalities. 
    All outputs should have a cell by gene matrix, a cell metadata dataframe, and 
    a transcript dataframe with transcript to cell allocations

    attributes: 
        image_dir : the directory of the image merscope outputs for a given region
        scale : the scale at which to load the images 
    """
    def __init__(self, cell_by_gene_path:str, cell_meta_path:str=None, cell_polygons_path:str=None,
                 seg_name:str="default", exp_name:str=None, reg_name:str=None, donor_name:str=None,
                 ):
        # args
        self.cbg_path = cell_by_gene_path
        self.meta_path = cell_meta_path
        self.polygons_path = cell_polygons_path
        
        # parameters for identifying the experiment
        self.seg_name = seg_name
        self.experiment = exp_name
        self.region = reg_name
        self.donor = donor_name
        self.load_preset(seg_name=seg_name)
        
        # load the cbg and cell meta dataframes
        self.cell_meta = self.load_cell_meta()
        self.cbg = self.load_cell_by_gene()
        if self.polygons_path is not None:
            self.cell_polygons = self.load_cell_polygons()

    def load_preset(self, seg_name:str="default"):
        """
        Load the preset for the segmentation algorithm
        """
        if seg_name == "proseg":
            self.preset = PROSEG_PRESET
        else:
            self.preset = DEFAULT_PRESET

    def load_cell_by_gene(self):
        #### TODO: add test cases
        """
        Load the cell by gene matrix 
        """
        cbg = pl.read_csv(self.cbg_path)
        if self.preset['tz_cell_id'] in cbg.columns:
            cbg = cbg.rename({self.preset['tz_cell_id']: CELL_ID})
        else: 
            cbg = cbg.with_columns(self.cell_meta.select(pl.col(CELL_ID)))
        return cbg
    
    def load_cell_meta(self):
        ### TODO: add test cases
        """
        Load the cell by gene matrix 
        """
        df_meta = pl.read_csv(self.meta_path)
        self.meta_columns = df_meta.columns
        self.num_cells = df_meta.shape[0]
        df_meta = df_meta.rename(self.preset['meta_map'])
        # PROSEG has repeat cell ids for some reason --> !!!
        # df_meta = df_meta.with_columns(pl.concat_str([pl.col(CELL_ID),
        #                                               pl.col(CELL_FOV),
        #                                             ], separator=".").alias(CELL_ID))
        return df_meta
    
    def load_cell_polygons(self): 
        """
        Load the cell polygons 
        """
        if self.polygons_path.endswith(".gz"): 
            with gzip.open(self.polygons_path, 'rt') as f:
                cell_polygons = gpd.read_file(f)
        elif self.polygons_path.endswith(".parquet"):
            cell_polygons = gpd.read_parquet(self.polygons_path)
        else: 
            cell_polygons = gpd.read_file(self.polygons_path)

        cell_polygons = cell_polygons.rename_geometry(GEOMETRY_COL)
        if self.seg_name == "default" or self.seg_name == "vpt": 
            cell_polygons = cell_polygons.loc[cell_polygons[self.preset['geom_depth_col']] == 3]
        return cell_polygons        
    
    def get_cell_by_gene(self) -> pl.DataFrame:
        """
        Get the cell by gene matrix
        """
        return self.cbg
    
    def get_cell_meta(self) -> pl.DataFrame:   
        """
        Get the cell metadata
        """
        return self.cell_meta
    
    def get_cell_polygons(self) -> gpd.GeoDataFrame:
        """
        Get the cell polygons
        """
        if self.cell_polygons is None:
            raise ValueError("Cell polygons have not been loaded")
        return self.cell_polygons
    
    def get_seg_name(self) -> str:
        """
        Get the segmentation name
        """
        return self.seg_name
    
    def add_class_info(self, df:pl.DataFrame) -> pl.DataFrame:
        """
        Add the class information
        """
        df = df.with_columns(
            experiment = pl.lit(self.experiment),
            segmentation = pl.lit(self.get_seg_name()),
            region = pl.lit(self.region),
            donor = pl.lit(self.donor),
        )
        df = df.with_columns(pl.concat_str([pl.col(CELL_ID),
                                            pl.col("experiment"),
                                            pl.col("region"),
                                            pl.col("segmentation"),
                                            ], separator=".").alias("Index"))
        return df
    
    def _get_cbg_feature(self, post_filtering:bool=False) -> pl.DataFrame:
        """
        Get summary feature of the cell by gene matrix
        attributes:
            - post_filtering: if True, only return the features for the cells that pass the filtering
            (self.qc_cells must be set for if this is true) 
        """
        if post_filtering and self.qc_cells is None:
            raise ValueError("Filtering has not been run, please run filtering before getting features with post_filtering=True")

        # separating out the true genes from the blank counts 
        is_blank = (np.char.startswith(self.cbg.columns, "Blank-"))
        blank_data = self.cbg[:, is_blank]
        gene_data = self.cbg[:, ~is_blank]
    
        # calculating features 
        df_feature = gene_data.select(pl.exclude(CELL_ID)).sum_horizontal().to_frame()
        df_feature = df_feature.with_columns(self.cbg.select(pl.col(CELL_ID)))
        df_feature = df_feature.rename({"sum": "nCount_RNA"})
        df_feature = df_feature.with_columns(nFeature_RNA=(self.cbg.select(pl.exclude(CELL_ID)) != 0).sum_horizontal())
        df_feature = df_feature.with_columns(blank_data.sum_horizontal().alias("nBlank"))
        # adding general info 
        df_feature = self.add_class_info(df_feature)

        if post_filtering: 
            df_feature = df_feature.filter(pl.col(CELL_ID).is_in(self.qc_cells))
        return df_feature
    
    def _get_meta_feature(self, post_filtering:bool = False) -> pl.DataFrame:
        """
        Get summary feature of the cell metadata
        attributes:
            - post_filtering: if True, only return the features for the cells that pass the filtering
            (self.qc_cells must be set for if this is true) 
        """
        if post_filtering and self.qc_cells is None:
            raise ValueError("Filtering has not been run, please run filtering before getting features with post_filtering=True")
        
        df_feature = self.cell_meta.select([CELL_ID, CELL_X, CELL_Y, CELL_FOV, CELL_VOLUME])
        df_feature = self.add_class_info(df_feature)
        
        if post_filtering: 
            df_feature = df_feature.filter(pl.col(CELL_ID).is_in(self.qc_cells))

        return df_feature

    def get_features(self, post_filtering:bool=False) -> pl.DataFrame:
        """
        Get the features of the cell by gene matrix and cell metadata
        attributes:
            - post_filtering: if True, only return the features for the cells that pass the filtering
            (self.qc_cells must be set for if this is true) 
        """
        df_join = self._get_cbg_feature(post_filtering).join(self._get_meta_feature(post_filtering), on="Index", how="inner", coalesce=True)
        # Adding the count / volume ratio per cell 
        df_join = df_join.with_columns(pl.col("nCount_RNA").truediv(pl.col("volume")).alias("nCount_RNA_per_Volume"))
        # return a list of specific genes (info)
        return df_join.select(['Index', CELL_ID, "experiment", "region", "segmentation", CELL_X, CELL_Y, CELL_FOV, CELL_VOLUME,
                                'nCount_RNA', 'nFeature_RNA', 'nBlank', 'nCount_RNA_per_Volume'])
    
    # def plot_QC_metrics(self, pdfFile=None, **kwargs): 
    #     """
    #     Plot the QC metrics for the current segmentation
    #     """
    #     df_feature = self.get_features()

    def get_gene_cbg(self) -> pl.DataFrame: 
        """
        Get the cell by gene matrix
        """
        # separating out the true genes from the blank counts 
        is_blank = (np.char.startswith(self.cbg.columns, "Blank-"))
        gene_data = self.cbg[:, ~is_blank]
        return gene_data
    
    def run_QC_filtering(self, 
                         volume_min=100, volume_max=4000, vol_max_mult=3, max_vol_by_median=True,
                         n_count_min=20, n_count_max=1000, 
                         n_gene_min=5, n_gene_max=1000,
                         n_blank_min=-1, n_blank_max=5,
                         ncpv_min_quantile=0.02, ncpv_max_quantile=0.98,
                         plot=True): 
        """
        Given the Cell by Gene and cell metadata matrix, run the standard QC algo, 
        """
        # Get the feature matrix
        df_feature = self.get_features()
        n_count_per_volume = df_feature['nCount_RNA_per_Volume'].to_numpy()

        # Calculate the filtering thresholds that are data dependent
        if max_vol_by_median: 
            volume_max = df_feature['volume'].median() * vol_max_mult
        qMax = np.quantile(n_count_per_volume, ncpv_max_quantile)
        qMin = np.quantile(n_count_per_volume, ncpv_min_quantile)
        n_count_per_volume_min = qMin
        n_count_per_volume_max = qMax

        # Apply the filtering 
        judge = df_feature.filter(
            pl.col("volume").is_between(volume_min, volume_max) & 
            pl.col("nCount_RNA").is_between(n_count_min, n_count_max) &
            pl.col("nFeature_RNA").is_between(n_gene_min, n_gene_max) &
            pl.col("nBlank").is_between(n_blank_min, n_blank_max) &
            pl.col("nCount_RNA_per_Volume").is_between(n_count_per_volume_min, n_count_per_volume_max)
            ).select(CELL_ID)

        df_feature = df_feature.with_columns(pl.col(CELL_ID).is_in(judge).alias("pass_qc"))
        # Save the filtering Results: 
        self.qc_cells = judge

        # Plot the Filtering: 
        if plot: 
            fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16, 8), constrained_layout=True)

            plot_feature_distribution(df_feature, feature="volume", feature_alias="Cell Volume",
                                    min_val=volume_min, max_val=volume_max, ax=axes[0][0])
            plot_feature_distribution(df_feature, feature="nCount_RNA", feature_alias="RNA Count",
                                    min_val=n_count_min, max_val=n_count_max, ax=axes[0][1])
            plot_feature_distribution(df_feature, feature="nFeature_RNA", feature_alias="Num Unique RNAs", 
                                    min_val=n_gene_min, max_val=None, ax=axes[0][2])
            plot_feature_distribution(df_feature, feature="nBlank", feature_alias="Blank Count", 
                                    min_val=None, max_val=n_blank_max, ax=axes[1][0]) 
            plot_feature_distribution(df_feature, feature="nCount_RNA_per_Volume", feature_alias="Transcripts Per Cell Size",
                                    min_val=n_count_per_volume_min, max_val=n_count_per_volume_max, ax=axes[1][1])
            plot_filt_cells(df_feature, ax=axes[1][2], title="Filtered Cells")
            plt.suptitle("QC_Summary - Exp %s; - Region %s; - Segmentation %s" %(self.experiment, self.region, self.seg_name))

            return fig, axes
        else:
            return None, None

    def gen_adata(self) -> ad.AnnData:
        """
        Generate an AnnData object from the cell by gene matrix and cell metadata
        """
        # Get the cell by gene matrix
        cbg = self.get_gene_cbg()
        cbg = cbg.filter(pl.col(CELL_ID).is_in(self.qc_cells))
        cbg = cbg.select(pl.exclude(CELL_ID))
        
        # Get the cell metadata
        df_feature = self.get_features()
        df_feature = df_feature.filter(pl.col(CELL_ID).is_in(self.qc_cells))
        
        # Create the AnnData object
        adata = ad.AnnData(X=sp.sparse.csr_matrix(cbg.to_numpy()),
                           obs=df_feature.to_pandas(),
                           var={"gene_name" : cbg.columns})
        adata.obs.set_index("Index", inplace=True)
        adata.var.set_index("gene_name", inplace=True)
        return adata
        
    def get_num_cells(self, post_filtering:bool=False) -> int: 
        """
        Return the number of cells in the segmentation 
        attributes:
            - post_filtering: if True, only return the features for the cells that pass the filtering
            (self.qc_cells must be set for if this is true)
        """
        if post_filtering and self.qc_cells is None:
            raise ValueError("Filtering has not been run, please run filtering before getting features with post_filtering=True")
    
        if post_filtering:
            return self.qc_cells.shape[0]
        return self.num_cells
    
    def cell_image_fraction(self): 
        """
        Return the fraction of the image occupied by cells
        """
        # TODO:
        # Calculate total image volume 
        total_image_volume = 1
        total_cell_volume = self.cell_meta.select("volume").sum()
        return total_cell_volume / total_image_volume
    
    def shilouette_score(self): 
        """
        Calculate scores based on shilouette scores: 
        - Need to calculate clusters (KNN is the most straight forward w/ varying k values)
        - Over all K calculate a CV score + variance explained by first PC + shilouette score (from sklearn)
        """
        ### TODO 
        return (0)

    def tr_per_cell(self, plot=False): 
        """
        Calculate the average tz / cell 
        If plotting do the violin plot of the dist. 
        """
        ### TODO Code this in the Transcript Class
        # Add a case for if there is a background segmentation add that as well. 
        return 0 

    def cells_per_100(self, plot=False):
        """
        Calculate the numebr of cells per 100um squared 
        if plot also plot the dist. 
        """
        # TODO 
        return 0 

    def cell_size_sd(self, post_filtering:bool=False):
        """
        Calculate the variation in cell sizes
        upper bounded at 1, which is same cell size
        attributes: 
            - post_filtering: if True, only return the features for the cells that pass the filtering
            (self.qc_cells must be set for if this is true)
        """
        if post_filtering and self.qc_cells is None:
            raise ValueError("Filtering has not been run, please run filtering before getting features with post_filtering=True")
        
        if post_filtering:
            cell_volumes = self.cell_meta.filter(pl.col(CELL_ID).is_in(self.qc_cells)).select("volume")
        else: 
            cell_volumes = self.cell_meta.select("volume")
        return 1 / (np.log(cell_volumes.std()).flatten() + 1)

        