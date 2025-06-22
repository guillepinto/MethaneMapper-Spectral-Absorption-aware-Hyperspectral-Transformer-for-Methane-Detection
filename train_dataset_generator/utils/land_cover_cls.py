#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:50:48 2022.

@author: satish
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Indexes:
    def __init__(self, args, band_centers=None):
        self.args = args
        print("Initializing Index computation class")
        self.epsilon = 0.0000001
        self.mask_val = -100.0

        if band_centers is None:
            # Default band centers for AVIRIS-NG (approximate)
            self.band_centers = {"blue": 15, "green": 31, "red": 57, "nir": 88}
        else:
            # User-provided band centers for other sensors (e.g., EnMap)
            self.band_centers = band_centers

        print(f"Using band centers: {self.band_centers}")

    def _get_band(self, img, center_idx, std_dev=3, num_samples=10):
        """
        Helper function to extract an average band value around a center index.
        This simulates a sensor's response for a particular color.
        """
        max_band_idx = img.shape[2] - 1
        # Generate band indices from a normal distribution around the center
        idxs = np.random.normal(center_idx, std_dev, num_samples)
        # Ensure indices are integers and within the valid range of the image bands
        idxs = np.clip(np.round(idxs), 0, max_band_idx).astype(int)
        band_data = img[:, :, idxs]
        # Return the mean value of the selected bands
        return np.mean(band_data, axis=2)

    def ndvi(self, img):
        # Calculate the average values for Blue, Green, Red, and NIR bands
        # using the configured band centers.
        self.B = self._get_band(img, self.band_centers["blue"])
        self.G = self._get_band(img, self.band_centers["green"])
        self.R = self._get_band(img, self.band_centers["red"])
        self.Infra = self._get_band(img, self.band_centers["nir"])

        # Allow division by zero
        np.seterr(divide="ignore", invalid="ignore")
        """
				#water
				ndvi_water = (G.astype(float) - infra.astype(float))/(infra+G)
				if(np.isnan(ndvi_water).any()):
						ndvi_water = np.ma.masked_invalid(ndvi_water)
				"""

        # vegetation
        ndvi_veg = (self.Infra.astype(float) - self.R.astype(float)) / (self.Infra + self.R + self.epsilon)
        # if(np.isnan(ndvi_veg).any()):
        # 	ndvi_veg = np.ma.masked_invalid(ndvi_veg)

        return ndvi_veg

    def getLandCls(self, img, img_mask):
        img = self.ndvi(img)
        img[img_mask] = self.mask_val
        ndvi_class_bins = [-np.inf, self.mask_val + 1, -0.6, -0.3, -0.08, 0, 0.4, 0.6, 0.8, np.inf]
        ndvi_class = np.digitize(img, ndvi_class_bins)
        cus_clr = ["black", "red", "orange", "salmon", "y", "olive", "yellowgreen", "g", "darkgreen"]
        self.cus_cmap = ListedColormap(cus_clr)
        self.ndvi_class = np.ma.masked_where(np.ma.getmask(img), ndvi_class)

        # generate RGB color image for tiling
        clr_img = self.visualize_rgb(split_flag=True, mask=img_mask)

        return self.cus_cmap, self.ndvi_class, clr_img

    def visualizer(self):
        if self.args.side_by_side:
            self.visualize_rgb(split_flag=False)
            self.visualize_land_cls()
            self.visualize_side_by_side()
        else:
            if self.args.color_img:
                self.visualize_rgb(split_flag=False)
            if self.args.visualize:
                self.visualize_land_cls()

    def visualize_rgb(self, split_flag=False, mask=None):
        # create color image of ground terrain
        B = self.B
        G = self.G
        R = self.R

        if mask is not None:
            B = np.ma.masked_array(B, mask=mask)
            G = np.ma.masked_array(G, mask=mask)
            R = np.ma.masked_array(R, mask=mask)

        def normalize_band(band):
            band_min = band.min()
            band_max = band.max()
            # Avoid division by zero if the band is flat
            if band_max == band_min:
                return np.zeros_like(band.data, dtype=np.float32)
            return ((band.astype(np.float32) - band_min) / (band_max - band_min)) * 255

        R_norm = normalize_band(R)
        G_norm = normalize_band(G)
        B_norm = normalize_band(B)

        # Fill masked values with 0 (black) before stacking
        # Stack in R, G, B order for a standard RGB image
        clr_img = np.uint8(np.stack((R_norm.filled(0), G_norm.filled(0), B_norm.filled(0)), axis=2))
        
        print("Equalizing rgb histogram")
        
        # Convert to YUV to equalize luminance (Y channel) and preserve color
        img_yuv = cv2.cvtColor(clr_img, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        clr_img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        if not split_flag:
            # imwrite expects BGR, so convert from RGB before saving
            cv2.imwrite("rgb_out.png", cv2.cvtColor(clr_img_equalized, cv2.COLOR_RGB2BGR))
            print("RGB image generator")
        else:
            return clr_img_equalized

    def visualize_land_cls(self):
        vmin = self.ndvi_class.min()
        vmax = self.ndvi_class.max()

        plt.imsave("out.png", self.ndvi_class, vmin=vmin, vmax=vmax, cmap=self.cus_cmap)

    def visualize_side_by_side(self):
        rgb_im = cv2.imread("rgb_out.png")
        cls_im = cv2.imread("out.png")
        sbs_img = cv2.hconcat([rgb_im, cls_im])
        cv2.imwrite("sbs_out.png", sbs_img)
