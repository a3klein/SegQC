# system
import os
import sys
import json
from pathlib import Path

import collections
import polars as pl

# data 
import numpy as np
import tifffile 
from skimage.filters import threshold_otsu

class Stains(): 
    """
    A class to handle image loading and processing.
    Specific to MERSCOPE image output format 

    attributes: 
        image_dir : the directory of the image merscope outputs for a given region
        scale : the scale at which to load the images 
    """
    def __init__(self, image_dir, scale:int=1):
        # args
        self.image_dir = image_dir
        self.image_scale = scale  
        # path files 
        self.transform_path = f"{self.image_dir}/micron_to_mosaic_pixel_transform.csv"
        self.manifest_path = f"{self.image_dir}/manifest.json"
        # Loading the images 
        self.image_dict = self.load_images()
        # Loading the additional information
        self.um_to_px, self.px_to_um = self._create_rescale_function(self.transform_path)
        self.bbox_microns = self._read_manifest(self.manifest_path)

        self.num_channels = len(self.image_dict.keys())

    def get_num_channels(self):
        return self.num_channels
    # def set_num_channels(self, a): 
    #     self.num_channels = a
    # def del_num_channels(self): 
    #     del self.num_channels
    
    # num_channels = property(
    #     fget=get_num_channels,
    #     fset=set_num_channels,
    #     fdel=del_num_channels,
    #     doc="The Number of Channels",
    # )
    
    def load_images(self) -> collections.defaultdict:
        """
        Load all images in the directory
        """
        image_dict = collections.defaultdict(dict)

        image_files = list(Path(self.image_dir).glob('*.tif'))
        self.image_files = {f.stem.split('_')[1]: f for f in image_files}
        
        # load all images 
        for name, filepath in self.image_files.items():
            image = self._load_single_image(filepath, scale=self.image_scale)
            # save the image
            print(f"Loaded {name} with shape {image.shape}")
            image_dict[name] = image

        return image_dict
    
    # For loading a single stain image at a given scale 
    def _load_single_image(self, filepath, scale:int=1) -> np.array:
        # using tiffile to open image
        with tifffile.TiffFile(filepath) as tif: 
            page = tif.pages[0]
            height = page.imagelength
            width = page.imagewidth
            # calculating padding & shape based on scale
            pad_height = (scale - height % scale) % scale
            pad_width = (scale - width % scale) % scale
            # final shape of image to load
            shape = (height + pad_height) // scale, (width + pad_width) // scale
            # load the image
            stack = np.empty(shape, page.dtype)
            stack = page.asarray(out='memmap')[::scale, ::scale]
        return stack
        
    # For the micron mosaic transformation 
    def _create_rescale_function(self, filepath) -> tuple: 
        """

        """
        um_to_px = np.loadtxt(filepath)
        px_to_um = np.linalg.inv(um_to_px)
        return (um_to_px, px_to_um)

    # For getting bounding box of the image
    def _read_manifest(self, filepath) -> np.array: 
        """

        """
        with open(filepath, 'r') as f: 
            manifest = json.load(f)
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = manifest['bbox_microns']
        return np.asarray([(bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax)])
    
    
    #
    def get_channels(self) -> list :
        """
        For getting a list of the available channels
        """
        return list(self.image_dict.keys())

    # 
    def get_image(self, image_name) -> np.array: 
        """
        Wrapper for _get_image to convert to numpy array (makes a deep copy)
        """
        return np.asarray(self._get_image(image_name))
    
    # 
    def _get_image(self, image_name) -> np.array : 
        """
        For Loading an image in memmap format
        """
        if image_name in self.image_dict.keys(): 
            return self.image_dict[image_name]
        else: 
            raise ValueError(f"""Image name not found in image dict. Image passes: {image_name}; Images available: {self.get_channels()} """)
        
    # 
    def image_total_intensity(self, image_name) -> int: 
        """
        Getting the total intensity of an image
        """ 
        return self._get_image(image_name).sum()
    
    def signal_to_noise_Z(self, image_name) -> float: 
        """
        The zscore signal to noise ratio is the ratio of the mean intensity to the std of intensity
        """
        img = self._get_image(image_name)
        return img.mean() / img.std()
    
    def signal_to_noise_Otsu(self, image_name) -> float: 
        """
        The Otsu signal to noise ratio is the mean intensity  above the otsu threshold to the mean intensity below it 
        """
        img = self._get_image(image_name)
        thr = threshold_otsu(img)
        return img[img > thr].mean() / img[img < thr].mean()
    
    # def signal_to_noise(self) -> collections.defaultdict: 
    #     res = collections.defaultdict(dict)
    #     for img_name in self.image_dict.keys(): 
    #         res[img_name] = {}
    #         res[img_name]['S/N - Z'] = self.signal_to_noise_Z(img_name)
    #         res[img_name]['S/N - O'] = self.signal_to_noise_Otsu(img_name)
    #     return res
    
    def _gen_single_stain_report(self, img_name) -> list: 
        """
        Generating a report (a list of values) for a single image
        -- ret order [stain, intensity, S/N Z, S/N O]
        """
        return [img_name,
                self.image_total_intensity(img_name),
                self.signal_to_noise_Z(img_name),
                self.signal_to_noise_Otsu(img_name),
            ]

    def gen_image_report(self):
        """
        Iterating over all images and generating the report for all of them
        """
        rep = []
        for img_name in self.image_dict.keys(): 
            rep.append(self._gen_single_stain_report(img_name))
        return pl.DataFrame(rep, schema=['Stain', 'Intensity', 'S/N - Z', 'S/N - O'], orient="row")
        
        
        

        