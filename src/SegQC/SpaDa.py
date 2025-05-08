
# system
import os
import sys

# data 
import numpy as np
import polars as pl
import geopandas as gpd
import anndata as ad

# plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

seg_qc_path = "/ceph/cephatlas/aklein/SegQC/src"
sys.path.append(seg_qc_path)
from SegQC.Stains import Stains
from SegQC.SegMask import SegMask
from SegQC.Transcripts import Transcripts
from SegQC.plotting import stain_plots, transcript_plots, segmentation_plots


class SpatialData(): 
    """
    A class for storing a spatial dataset and a segmentation mask.

    attributes: 
        transctripts : transcript location files (.csv / .parquet file)
        image : image files for intensity (.tif files)
        segmentation : a segmentation mask (.csv / .parquet file)
    """
    def __init__(self,
                 transcripts_path:str=None,
                 image_path:str=None,
                 reg_name:str=None,
                 exp_name:str=None, 
                 donor_name:str=None,
                 segmentation_path:str=None, # There is no really one path to segmentation object
                ): 
        """
        Initialize the SpatialData object with paths to the transcripts, image, and segmentation files.

        Args:
            reg_name (str) : The name of the region being analyzed
            transctripts_path (str): Path to the transcripts file.
            image_path (str): Path to the image files.
            segmentation_path (str): Path to the segmentation mask file.
        """
        self.reg_name=reg_name
        self.exp_name=exp_name
        self.donor_name=donor_name

        self.transcripts_path = transcripts_path
        self.image_path = image_path

        # self.segmentation_path = segmentation_path
    
    def load_images(self): 
        """
        Load the Stains object
        """
        self.stains = Stains(self.image_path, scale=10) # TODO change scale back to 1 once done with testing

    def load_transcripts(self):
        """
        Load the Transcripts object
        """
        self.transcripts = Transcripts(self.transcripts_path)

    def load_segmentation(self,
                          cell_by_gene_path:str,
                          cell_meta_path:str=None,
                          transcripts_path:str=None,
                          cell_polygons_path:str=None,
                          segmentation_type:str="default",
                          ):
        """
        Load the segmentation mask data.
        attributes: 
            cell_by_gene_path (str): Path to the cell by gene matrix file.
            cell_meta_path (str): Path to the cell metadata file.
            transcripts_path (str): Path to the transcripts file with annotated assignmets. 
                If this file is supplied it will override the existing Transcript object in this class
        """
        self.segmentation = SegMask(cell_by_gene_path=cell_by_gene_path,
                                    cell_meta_path=cell_meta_path,
                                    cell_polygons_path=cell_polygons_path,
                                    seg_name=segmentation_type,
                                    exp_name=self.exp_name,
                                    reg_name=self.reg_name,
                                    donor_name=self.donor_name,)
        if transcripts_path:
            self.transcripts = Transcripts(transcripts_path, file_orig=segmentation_type)
        
    def get_transcripts(self) -> pl.DataFrame:
        """
        Get the transcript data.

        Returns:
            pl.DataFrame: A DataFrame containing the transcript data.
        """
        return self.transcripts
    
    def get_segmentation(self) -> gpd.GeoDataFrame:
        """
        Get the segmentation mask data.

        Returns:
            pl.DataFrame: A DataFrame containing the segmentation mask data.
        """
        return self.segmentation
    
    def get_stains(self) -> Stains: 
        """
        Get the Stains object

        Returns:
            Stains: A Stains object with the images for this dataset
        """
        return self.stains
        
    def read_segmentation_mask(self) -> pl.DataFrame:
        """
        Read the segmentation mask data from the specified path.

        Returns:
            pl.DataFrame: A DataFrame containing the segmentation mask data.
        """
        if self.segmentation_path.endswith('.csv'):
            return gpd.read_csv(self.segmentation_path)
        elif self.segmentation_path.endswith('.parquet'):
            return gpd.read_parquet(self.segmentation_path)
        else:
            raise ValueError("Unsupported file format for segmentation mask. Please provide a .csv or .parquet file.")
        
    def run_presegmentation_report(self, 
                                   plot=False,
                                   pdf_path:str=None, 
                                   plot_report_table=False, 
                                   control_genes:list=None, 
                                   **kwargs):     
        # if pdf_path and not plot: 
        #     raise ValueError("If pdf_path is given, plot must be True")
        
        stains_report = self.stains.gen_image_report()
        tz_report = self.transcripts.gen_transcripts_report()
        
        if plot: 
            img_plot_fig = stain_plots.plot_stains(stains=self.stains, 
                                                reg_name = self.reg_name,
                                                plot_report = plot_report_table)
            tz_plot_fig, _ = stain_plots.plot_tz_report(Tr=self.transcripts, 
                                                    reg_name = self.reg_name, 
                                                    control_genes=control_genes)
            tz_pipeilne_plot_fig, _ = transcript_plots.plot_base_pipeline(self.transcripts, **kwargs)

            # if saving to a pdf
            if pdf_path: 
                pdf_file = PdfPages(pdf_path)
                pdf_file.savefig(img_plot_fig, bbox_inches="tight", dpi=300)
                pdf_file.savefig(tz_plot_fig, bbox_inches="tight", dpi=300)
                pdf_file.savefig(tz_pipeilne_plot_fig, bbox_inches="tight", dpi=300)
                pdf_file.close()
                plt.close(img_plot_fig)
                plt.close(tz_plot_fig)
                plt.close(tz_pipeilne_plot_fig)
    
            else: 
                # if not saving to the pdf, show the figures
                plt.show(tz_pipeilne_plot_fig)
                plt.show(tz_plot_fig)
                plt.show(img_plot_fig)
        
        return (stains_report, tz_report)
        # TODO: make the report a 2C polars table with "index" the category
    
    def run_QC_filtering(self, plot:bool=False, pdf_path:str=None, **seg_kwargs): 
        """
        # ## FILTERING PARAMETERS  Need to be passed in as args? 
            volume_min = 30
            max_vol_by_median = True
            vol_max_mult = 3
            volume_max = 4000
            # XZ had volume > 3 * median volume instead of this max value here
            n_count_min = 20 # 20
            n_count_max = 1000
            ncpv_min_quantile = 0.02
            ncpv_max_quantile = 0.98
            n_gene_min = 5 # 5
            n_blank_max = 5
        """
        fig, ax = self.segmentation.run_QC_filtering(plot=plot, **seg_kwargs)
        if plot: 
            if pdf_path: 
                pdf_file = PdfPages(pdf_path)
                pdf_file.savefig(fig, bbox_inches="tight", dpi=300)
                pdf_file.close()
                plt.close(fig)
            else: 
                plt.show(fig)        
                plt.close(fig) 
        adata = self.segmentation.gen_adata()
        return adata
        
    def run_segmentation_QC(self, post_filtering:bool = True) -> pl.DataFrame:
        """
        Generate a segmentation QC report (either pre / post filtering)
        """

        # generate more specific segmentation statistics
        df_stats = pl.DataFrame([[self.segmentation.get_num_cells(post_filtering),
                                  self.segmentation.cell_size_sd(post_filtering),
                                  self.transcripts.ratio_tz_in_cells(self.segmentation.qc_cells)],
                                  ['num_cells', '1/SD(cell_size)', '%_tz_in_cells']],
                                schema={"value":pl.Float64, "index":str})
        
        # Read in the features of the segmentation
        df_feats = self.segmentation.get_features(post_filtering=post_filtering)
        # generate descriptive statistics for each feature 
        df_desc = df_feats['nCount_RNA', 'nFeature_RNA', 'volume', 'nCount_RNA_per_Volume'].describe()
        df_desc = df_desc.filter(~pl.col("statistic").is_in(['count', 'null_count'])).unpivot(index="statistic")
        df_desc = df_desc.with_columns((pl.col("variable") + "-" + pl.col("statistic")).alias("index")).drop("statistic", "variable")
        
        # add the more general segmentation statistics
        df_desc = df_desc.vstack(df_stats)

        return df_desc
    

    def plot_fov(self, stain_name:str="DAPI", bbox:list=[3000,3000,3500,3500],
                 pdfFile:PdfPages=None, plot:bool=True, figsize=(15,15),**kwargs):
        """
        Plot the field of view for the specified stain.
        attributes: 
            stain_name (str): The name of the stain to plot.
            bbox (list): The bounding box for the field of view.
            scale_factor (int): The scale factor for the image.
            pdf_path (str): The path to save the PDF file.
            plot (bool): Whether to plot the figure or not.
        """

        img = self.stains.get_image(stain_name)
        assert(bbox[0] >= 0, bbox[1] >= 0) 
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax = stain_plots.plot_fov(img, bbox, ax=ax)
        ax = segmentation_plots.plot_cell_seg_fov(self.segmentation.get_cell_polygons(), mmpt=self.stains.um_to_px, 
                                                  bbox=bbox, scale_factor=self.stains.image_scale, ax=ax)
        ax.set_title(f"{self.reg_name} - {stain_name} - {self.segmentation.seg_name} - {bbox}")
        plt.gca().set_aspect('equal')
        if plot: 
            if pdfFile: 
                pdfFile.savefig(fig, bbox_inches="tight", dpi=300)
                plt.close(fig)
            else: 
                plt.show(fig)
                plt.close()