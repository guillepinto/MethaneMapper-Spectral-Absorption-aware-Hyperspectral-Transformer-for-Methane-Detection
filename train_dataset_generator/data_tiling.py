import os
import numpy as np
import math
import shutil
import cv2
import rasterio  
import argparse


class DataTiling:
    def __init__(self, size=256, offset=128):
        print("Init DataTiling class")
        self.tile_size = (size, size)
        self.offset = (offset, offset)

    def dataTiling(self, img, filename, dir_path, filetype=None):
        # create tiles directory for the current file
        _dir_path = f"{dir_path}/{filename}_tiles"
        self.createDirectory(_dir_path)

        img_shape = img.shape
        for i in range(int(math.ceil(img_shape[0] / (self.offset[1] * 1.0)))):
            for j in range(int(math.ceil(img_shape[1] / (self.offset[0] * 1.0)))):
                tile = img[
                    self.offset[1] * i : min(self.offset[1] * i + self.tile_size[1], img_shape[0]),
                    self.offset[0] * j : min(self.offset[0] * j + self.tile_size[0], img_shape[1]),
                ]

                if filetype == "png":
                    tile_name = f"{filename}_{i}_{j}.png"
                else:
                    tile_name = f"{filename}_{i}_{j}.npy"

                # Ensure tile is not empty before saving
                if tile.shape[0] > 0 and tile.shape[1] > 0:
                    if filetype == "png":
                        cv2.imwrite(os.path.join(_dir_path, tile_name), tile)
                    else:
                        np.save(os.path.join(_dir_path, tile_name), tile)

    def createDirectory(self, _dir_path):
        if os.path.isdir(_dir_path):
            print(f"Directory '{_dir_path}' already exists. Removing and recreating.")
            shutil.rmtree(_dir_path)
        os.makedirs(_dir_path)
        print(f"Created directory: '{_dir_path}'")


def main(input_dir, destination_dir, file_type):
    """
    Reads images from an input directory, tiles them, and saves them to the destination directory.
    Handles .png, .npy, and 4-channel GeoTIFF (.tif, .tiff) files.
    For GeoTIFFs, it assumes the 4th channel is the mask to be tiled.
    """
    # Ensure the main destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    supported_extensions = (".png", ".npy", ".tif", ".tiff")
    all_files = [f for f in os.listdir(input_dir) if f.endswith(supported_extensions)]
    all_paths = [os.path.join(input_dir, _name) for _name in all_files]

    if not all_paths:
        print(f"No supported files found in '{input_dir}'. Please add your files and run again.")
        return

    # initializing DataTiling class
    DTObj = DataTiling()

    for _path in all_paths:
        print(f"\nProcessing {_path}")
        img = None
        filename = os.path.basename(_path).split(".")[0]
        try:
            if _path.endswith((".tif", ".tiff")):
                with rasterio.open(_path) as src:
                    # For GeoTIFFs, we assume we want the 4th channel as a mask
                    if src.count == 4:
                        img = src.read(4)
                    else:
                        print(f"Warning: GeoTIFF {_path} has {src.count} channels, but expected 4 to extract a mask. Skipping.")
                        continue
            elif _path.endswith(".png"):
                # Use IMREAD_UNCHANGED to handle images with an alpha channel correctly
                img = cv2.imread(_path, cv2.IMREAD_UNCHANGED)
            elif _path.endswith(".npy"):
                img = np.load(_path)

            if img is not None:
                DTObj.dataTiling(img, filename, destination_dir, filetype=file_type)

        except Exception as e:
            print(f"Error processing {_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile image files from a directory. Handles .png, .npy, and 4-channel GeoTIFFs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input_data",
        help="The directory where your input files (.png, .npy, .tif) are located.",
    )
    parser.add_argument(
        "--destination_dir",
        type=str,
        default="../data/all_gt_masks",
        help="The destination directory for the tiled masks.",
    )
    parser.add_argument(
        "--file_type",
        type=str,
        choices=["png", "npy"],
        default="png",
        help="The file type for the output tiles.",
    )

    args = parser.parse_args()

    # Create a placeholder input directory if it doesn't exist, to prevent errors.
    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir)
        print(f"Created placeholder input directory: {args.input_dir}")
        print("Please place your input files in this directory and run the script again.")
    else:
        main(args.input_dir, args.destination_dir, args.file_type)
