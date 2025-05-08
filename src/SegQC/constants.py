import os
import sys
import pathlib
import numpy as np
import polars as pd
import collections


# Constants for column names
X_LOC = "X_GLOBAL"
Y_LOC = "Y_GLOBAL"
Z_LOC = "Z_GLOBAL"
CELL_ID = "CELL_ID"
GENE_NAME = "GENE_NAME"
GENE_ID = "GENE_ID"
FOV_COL = "FOV"

CELL_X = "CENTER_X"
CELL_Y = "CENTER_Y"
CELL_FOV = "fov"
CELL_VOLUME = "volume"
GEOMETRY_COL = "GEOMETRY"


DEFAULT_PRESET = {"tz_cell_id": "cell", 
                  "meta_map" : {"EntityID" : CELL_ID,
                                "fov" : CELL_FOV,
                                "center_x" : CELL_X,
                                "center_y" : CELL_Y,
                                "volume" : CELL_VOLUME,
                                },                
                "tz_col_mapper" : {"global_x":X_LOC,
                                   "global_y":Y_LOC,
                                   "global_z":Z_LOC,
                                   "cell_id":CELL_ID,
                                   "gene":GENE_NAME,
                                   "transcript_id":GENE_ID,
                                   "fov":FOV_COL,},
                "geom_depth_col" : "ZIndex",                
                }

PROSEG_PRESET = {"tz_cell_id" : None,
                 "meta_map" : {"cell" : CELL_ID,
                                "fov" : CELL_FOV,
                                "centroid_x" : CELL_X,
                                "centroid_y" : CELL_Y,
                                "volume" : CELL_VOLUME,
                                },
                "tz_col_mapper" : {"x":X_LOC,
                                   "y":Y_LOC,
                                   "z":Z_LOC,
                                   "assignment":CELL_ID,
                                   "gene":GENE_NAME,
                                   "transcript_id":GENE_ID,
                                   "fov":FOV_COL,},
                "geom_depth_col" : "layer",                 
                }

