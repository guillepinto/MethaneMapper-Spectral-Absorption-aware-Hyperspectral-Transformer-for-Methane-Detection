import os
import numpy as np
import math
import shutil
import cv2
import rasterio
import random


class DataTiling:
    def __init__(self, size=256, offset=128):
        print("Init DataTiling class")
        self.tile_size = (size, size)
        self.offset = (offset, offset)

    def generate_balanced_tiles(self, mask_img, rgb_img, filename, dir_path, filetype=None, balance_ratio=1.0):
        """
        Generates a balanced set of tiles.
        It saves all tiles containing a mask and a random sample of tiles without a mask.
        """
        _dir_path = f"{dir_path}/{filename}_tiles"
        self.createDirectory(_dir_path)

        img_shape = mask_img.shape
        tiles_with_mask = []
        tiles_without_mask = []

        print("Identifying tiles with and without masks...")
        for i in range(int(math.ceil(img_shape[0] / (self.offset[1] * 1.0)))):
            for j in range(int(math.ceil(img_shape[1] / (self.offset[0] * 1.0)))):
                y_start = self.offset[1] * i
                y_end = min(y_start + self.tile_size[1], img_shape[0])
                x_start = self.offset[0] * j
                x_end = min(x_start + self.tile_size[0], img_shape[1])

                mask_tile = mask_img[y_start:y_end, x_start:x_end]
                
                if mask_tile.shape[0] == 0 or mask_tile.shape[1] == 0:
                    continue

                tile_name = f"{filename}_{i}_{j}.{filetype}"
                tile_info = {'tile': mask_tile, 'name': tile_name}

                # Check if the mask tile has any plume pixels
                if np.sum(mask_tile) > 0:
                    tiles_with_mask.append(tile_info)
                else:
                    # Check if the corresponding RGB tile has image data (is not black)
                    rgb_tile = rgb_img[:, y_start:y_end, x_start:x_end]
                    if np.sum(rgb_tile) > 0:
                        tiles_without_mask.append(tile_info)
        
        print(f"Found {len(tiles_with_mask)} tiles with masks.")
        print(f"Found {len(tiles_without_mask)} valid tiles without masks.")

        # Save all tiles that have masks
        for tile_info in tiles_with_mask:
            self.save_tile(tile_info['tile'], tile_info['name'], _dir_path, filetype)

        # Balance with tiles that don't have masks
        num_without_mask_to_add = int(len(tiles_with_mask) * balance_ratio)
        
        # Ensure we don't try to sample more than we have
        num_to_sample = min(num_without_mask_to_add, len(tiles_without_mask))
        
        print(f"Sampling {num_to_sample} tiles without masks to balance the dataset.")
        
        if tiles_without_mask: # Check if list is not empty
            if num_to_sample < len(tiles_without_mask):
                sampled_tiles = random.sample(tiles_without_mask, num_to_sample)
            else:
                sampled_tiles = tiles_without_mask # Use all if not enough
        else:
            sampled_tiles = []


        for tile_info in sampled_tiles:
            self.save_tile(tile_info['tile'], tile_info['name'], _dir_path, filetype)
            
        total_saved = len(tiles_with_mask) + len(sampled_tiles)
        print(f"Saved a total of {total_saved} tiles to '{_dir_path}'.")


    def save_tile(self, tile, tile_name, dir_path, filetype):
        if filetype == "png":
            cv2.imwrite(os.path.join(dir_path, tile_name), tile)
        else:
            np.save(os.path.join(dir_path, tile_name), tile)

    def createDirectory(self, _dir_path):
        if os.path.isdir(_dir_path):
            print(f"Directory '{_dir_path}' already exists. Removing and recreating.")
            shutil.rmtree(_dir_path)
        os.makedirs(_dir_path)
        print(f"Created directory: '{_dir_path}'")


def main(geotiff_input_dir, destination_dir):
    """
    Reads 4-channel GeoTIFFs, extracts the 4th channel (mask), and saves a balanced
    set of mask tiles as PNG files.
    """
    os.makedirs(destination_dir, exist_ok=True)

    all_geotiff_files = [f for f in os.listdir(geotiff_input_dir) if f.endswith((".tif", ".tiff"))]
    all_geotiff_paths = [os.path.join(geotiff_input_dir, _name) for _name in all_geotiff_files]

    if not all_geotiff_paths:
        print(f"No GeoTIFF files found in '{geotiff_input_dir}'. Please add your files and run again.")
        return

    DTObj = DataTiling()

    for _geotiff_path in all_geotiff_paths:
        print(f"\nProcessing {_geotiff_path}")
        try:
            with rasterio.open(_geotiff_path) as src:
                if src.count == 4:
                    # Read RGB channels to check for valid image data
                    rgb_img = src.read((1, 2, 3))
                    # Read the 4th channel (the mask)
                    mask_img = src.read(4)
                    
                    filename = os.path.basename(_geotiff_path).split(".")[0]
                    DTObj.generate_balanced_tiles(mask_img, rgb_img, filename, destination_dir, filetype="png")
                else:
                    print(f"Warning: {_geotiff_path} has {src.count} channels, expected 4. Skipping.")
        except Exception as e:
            print(f"Error processing {_geotiff_path}: {e}")


if __name__ == "__main__":
    # --- Instructions ---
    # 1. Create a folder in your project, for example, 'geotiff_input'.
    # 2. Place your 4-channel GeoTIFF files inside this folder.
    # 3. Update the `GEOTIFF_INPUT_DIR` variable below to point to that folder.
    # 4. Run this script. The output tiles will be placed in `../data/all_gt_masks/`.

    # The directory where your 4-channel .tif files are located.
    # PLEASE UPDATE THIS PATH
    GEOTIFF_INPUT_DIR = "./geotiff_input"

    # The destination directory for the tiled masks.
    DESTINATION_DIR = "../data/all_gt_masks"

    # Create a placeholder input directory if it doesn't exist, to prevent errors.
    if not os.path.exists(GEOTIFF_INPUT_DIR):
        os.makedirs(GEOTIFF_INPUT_DIR)
        print(f"Created placeholder input directory: {GEOTIFF_INPUT_DIR}")
        print("Please place your 4-channel GeoTIFF files in this directory and run the script again.")

    main(GEOTIFF_INPUT_DIR, DESTINATION_DIR)