
# system
import os
import sys
import json
from pathlib import Path

# data 
import numpy as np
import polars as pl

# plotting
from SegQC.constants import DEFAULT_PRESET, PROSEG_PRESET
from SegQC.constants import X_LOC, Y_LOC, Z_LOC, CELL_ID, GENE_NAME, GENE_ID, FOV_COL

class Transcripts(): 
    def __init__(self, transcripts_path, file_orig="default"): 
        """
        Initializing a class to hold a transcripts object (either pre or post segmentation)
        attributes: 
            transcripts_path(str) : a path to the transcript file (either stores as csv or parquet)
            cell_id_col(str) : a string indicating the column name of the cell assignment is post segmentation
        """
        self.transcripts_path = transcripts_path
    
        if file_orig == "default":
            self.tr_df = self.read_transcripts(colname_mapper=DEFAULT_PRESET['tz_col_mapper'])
        elif file_orig == "proseg":
            self.tr_df = self.read_transcripts(colname_mapper=PROSEG_PRESET['tz_col_mapper'])
    
    def get_transcripts(self) -> pl.DataFrame:
        """
        Get the transcript data.

        Returns:
            pl.DataFrame: A DataFrame containing the transcript data.
        """
        return self.tr_df
    
    def read_transcripts(self, colname_mapper:dict=DEFAULT_PRESET['tz_col_mapper'],
                        ) -> pl.DataFrame: 
        """
        Read the transcript data from the specified path.

        attributes: 
            - the transformation between the column names in the file and the class standard

        Returns:
            pl.DataFrame: A DataFrame containing the transcript data.
            With the mapped gene names to the class standards
        """
        # colname_mapper = {xcol: X_LOC, ycol: Y_LOC, zcol: Z_LOC, cell_id_col: CELL_ID, gene_name_col: GENE_NAME, gene_id_col: GENE_ID, fov_col: FOV_COL}
        if self.transcripts_path.endswith('.csv') or self.transcripts_path.endswith('.csv.gz'):
            return pl.read_csv(self.transcripts_path).rename(colname_mapper)
        elif self.transcripts_path.endswith('.parquet'):
            return pl.read_parquet(self.transcripts_path).rename(colname_mapper)
        else:
            raise ValueError("Unsupported file format for transcripts. Please provide a .csv or .h5 file.")
        
    
    def calc_tr_cdf(self, plot=False):
        from sklearn.linear_model import LinearRegression
        tr_gene_vc = self.tr_df[GENE_NAME].value_counts()
        tr_gene_vc = tr_gene_vc.select(pl.col(GENE_NAME, "count").gather(pl.arg_sort_by("count")))
        
        # for the linear regression 
        Y = np.log(tr_gene_vc["count"].to_numpy())
        X = np.arange(len(Y)).reshape(-1, 1)
        model = LinearRegression().fit(X, Y)
        
        # for downstream calculations
        self.tr_gene_vc = tr_gene_vc
    
        # the R2 value
        return model.score(X, Y)    


    def calc_tr_balanced(self): 
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import gaussian_kde
        #  from scipy.stats import multivariate_normal
        
        tr_mean_pos = self.tr_df.group_by(GENE_NAME).agg(pl.col(X_LOC, Y_LOC).mean())
        self.tr_mean_pos = tr_mean_pos
        ss = StandardScaler()
        np_mean_pos = ss.fit_transform(tr_mean_pos[X_LOC, Y_LOC].to_numpy()).T
        cov_det = np.linalg.det(np.cov(np_mean_pos))
        return cov_det
        
    def calc_voxel_dist(self, roundto=10, filt_thr=10): 
        locs = self.tr_df.select(pl.col([X_LOC, Y_LOC]))
        locs = locs.with_columns(pl.col([X_LOC, Y_LOC])/roundto).cast(pl.Int32)
        voxel_dist = locs.group_by([X_LOC, Y_LOC]).len().select('len').to_numpy().flatten()
        voxel_dist = voxel_dist[voxel_dist > filt_thr]
        self.voxel_dist = voxel_dist
        return voxel_dist.mean(), voxel_dist.std(), voxel_dist.mean() / voxel_dist.std()

    
    def run_base_pipeline(self): 
        r2 = self.calc_tr_cdf()
        disp = self.calc_tr_balanced()
        vox_m, vox_s, vox_m2s = self.calc_voxel_dist()
        return [r2, disp, vox_m, vox_s, vox_m2s]
    
    def gen_transcripts_report(self): 
        return pl.DataFrame([self.run_base_pipeline()],
                            schema=['cdf_r2', 'dispersion', 'voxel_mean', 'voxel_std', 'voxel_S/N_Z'],
                            orient='row')

    def tz_per_cell(self): 
        """
        For calculating the transcript per cell
        """
        tz = self.tr_df.group_by(CELL_ID).agg(pl.col(GENE_NAME).count())
        tz = tz.rename({GENE_NAME: "num_genes"})
        return tz
    
    def unique_tz_per_cell(self): 
        """
        For calculating the unique transcript per cell
        """
        tz = self.tr_df.group_by(CELL_ID).agg(pl.col(GENE_NAME).n_unique())
        tz = tz.rename({GENE_NAME: "num_unique_genes"})
        return tz
    
    def ratio_tz_in_cells(self, qcd_cells:list=None): 
        """
        For calculating the number of transcripts in cells
        """
        if qcd_cells is not None: 
            return self.tr_df[CELL_ID].is_in(qcd_cells).sum() / self.tr_df.shape[0]
        background_cell = self.tr_df.group_by(CELL_ID).agg(pl.len()).max()[CELL_ID]
        return self.tr_df.filter(pl.col(CELL_ID) != background_cell).shape[0] / self.tr_df.shape[0] 