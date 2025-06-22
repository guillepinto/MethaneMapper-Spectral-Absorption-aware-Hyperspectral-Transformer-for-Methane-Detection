import numpy as np
import os
import subprocess
from pathlib import Path
import sys
import argparse
import rasterio

# This script assumes it's run from the methanemapper directory.
# We add the necessary subdirectories to the path to import their modules.
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
train_gen_dir = project_root / "train_dataset_generator"
train_gen_utils_dir = train_gen_dir / "utils"

sys.path.insert(0, str(train_gen_dir))
sys.path.insert(0, str(train_gen_utils_dir))
sys.path.insert(0, str(script_dir))

from land_cover_cls import Indexes
from match_filter import segmentation_match_filter
from arg_parser import get_train_gen_parser 

# Get the default arguments from the training generator parser without parsing the command line.
# This creates a namespace with default values (e.g., pxl_batch_size) needed by helper functions.
train_gen_parser = get_train_gen_parser()
train_gen_args = train_gen_parser.parse_args([])
# We need to provide a method for the segmentation match filter. Let's use 'row_wise' as a default.
train_gen_args.segmentation_mf = "row_wise"

# --- EnMap Wavelengths (as provided by user) ---
# These are used to find the correct band indices for RGB/NDVI calculation.
vnir_wavelengths = [
    418.416, 424.043, 429.457, 434.686, 439.757, 444.699, 449.539, 454.306, 459.031, 463.73,
    468.411, 473.08, 477.744, 482.411, 487.087, 491.78, 496.497, 501.243, 506.02, 510.828,
    515.672, 520.55, 525.467, 530.424, 535.422, 540.463, 545.551, 550.687, 555.873, 561.112,
    566.405, 571.756, 577.166, 582.636, 588.171, 593.773, 599.446, 605.193, 611.017, 616.923,
    622.92, 628.987, 635.112, 641.294, 647.537, 653.841, 660.207, 666.637, 673.131, 679.691,
    686.319, 693.014, 699.78, 706.617, 713.524, 720.501, 727.545, 734.654, 741.826, 749.06,
    756.353, 763.703, 771.108, 778.567, 786.078, 793.639, 801.248, 808.905, 816.608, 824.355,
    832.145, 839.976, 847.847, 855.757, 863.703, 871.683, 879.692, 887.729, 895.789, 903.87,
    911.968, 920.08, 928.204, 936.335, 944.47, 952.608, 960.748, 968.891, 977.037, 985.186,
    993.338
]
swir_wavelengths = [
    901.962, 911.572, 921.32, 931.204, 941.218, 951.361, 961.629, 972.017, 982.524, 993.145,
    1003.88, 1014.72, 1025.66, 1036.7, 1047.84, 1059.07, 1070.39, 1081.79, 1093.26, 1104.81,
    1116.43, 1128.11, 1139.84, 1151.62, 1163.44, 1175.31, 1187.2, 1199.11, 1211.05, 1223.0,
    1234.97, 1246.95, 1258.93, 1270.92, 1282.92, 1294.91, 1306.9, 1318.88, 1330.86, 1342.82,
    1354.76, 1366.69, 1378.6, 1390.48, 1449.43, 1461.11, 1472.74, 1484.34, 1495.89, 1507.4,
    1518.87, 1530.29, 1541.68, 1553.01, 1564.31, 1575.55, 1586.76, 1597.91, 1609.02, 1620.09,
    1631.11, 1642.08, 1653.0, 1663.87, 1674.7, 1685.47, 1696.2, 1706.87, 1717.5, 1728.08,
    1738.6, 1749.08, 1759.51, 1769.89, 1780.22, 1967.66, 1977.08, 1986.45, 1995.79, 2005.08,
    2014.33, 2023.54, 2032.71, 2041.83, 2050.92, 2059.96, 2068.97, 2077.93, 2086.86, 2095.74,
    2104.59, 2113.4, 2122.17, 2130.9, 2139.6, 2148.26, 2156.88, 2165.47, 2174.02, 2182.53,
    2191.01, 2199.46, 2207.86, 2216.24, 2224.58, 2232.89, 2241.16, 2249.4, 2257.61, 2265.79,
    2273.93, 2282.04, 2290.12, 2298.17, 2306.19, 2314.18, 2322.13, 2330.05, 2337.94, 2345.81,
    2353.64, 2361.44, 2369.21, 2376.95, 2384.66, 2392.34, 2400.0, 2407.62, 2415.21, 2422.78,
    2430.32, 2437.83, 2445.3
]


def get_args_parser():
    """Defines the command-line arguments for the EnMap processing script."""
    parser = argparse.ArgumentParser("EnMap Processing Pipeline for MethaneMapper")
    # --- Input Paths ---
    parser.add_argument("--vnir_path", type=str, required=True, help="Path to the EnMap VNIR data cube (.TIF file)")
    parser.add_argument("--swir_path", type=str, required=True, help="Path to the EnMap SWIR data cube (.TIF file)")
    parser.add_argument("--methane_signature_path", type=str, default="../data/gas_signature/methane_signature.txt", help="Path to the original methane signature (.txt file).")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained MethaneMapper model checkpoint (.pth).")
    parser.add_argument("--stats_file_path", type=str, default="./data/data_stats", help="Path to the directory with 'dataset_mean.npy' and 'dataset_std.npy'.")
    # --- Output Path ---
    parser.add_argument("--output_dir", default="./enmap_processing_output", help="Main directory to save all processed tiles and inference results.")
    # --- Tiling Parameters ---
    parser.add_argument("--tile_size", type=int, default=256, help="The size of the square tiles to process.")
    parser.add_argument("--overlap", type=int, default=128, help="The overlap between adjacent tiles.")
    return parser

def find_closest_band(wavelengths, target_wl):
    """Helper function to find the index of the closest band."""
    return np.argmin(np.abs(np.array(wavelengths) - target_wl))

def tile_generator(image, tile_size, overlap):
    """Generator function to yield tiles and their coordinates from a large image."""
    height, width, _ = image.shape
    stride = tile_size - overlap
    for r in range(0, height - overlap, stride):
        for c in range(0, width - overlap, stride):
            # Ensure the tile does not go out of bounds
            r_end = min(r + tile_size, height)
            c_end = min(c + tile_size, width)
            # Create a full-sized tile by padding if necessary
            tile = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
            tile[:r_end-r, :c_end-c] = image[r:r_end, c:c_end]
            yield tile, (r, c)

def main(args):
    """Main processing pipeline for EnMap data."""
    print("--- Starting EnMap Data Processing for MethaneMapper ---")

    # --- 1. Load and Combine Data using Rasterio ---
    print("Step 1: Loading and combining VNIR and SWIR data...")
    if not (os.path.exists(args.vnir_path) and os.path.exists(args.swir_path)):
        print(f"ERROR: Input data not found. Please check --vnir-path and --swir-path arguments.")
        return

    try:
        with rasterio.open(args.vnir_path) as src:
            vnir_cube = src.read().transpose(1, 2, 0)  # Transpose to (H, W, Bands)
        with rasterio.open(args.swir_path) as src:
            swir_cube = src.read().transpose(1, 2, 0)
    except Exception as e:
        print(f"ERROR: Failed to read TIF files with rasterio: {e}")
        return

    # Combine cubes along the spectral (band) axis
    full_hsi_cube = np.concatenate((vnir_cube, swir_cube), axis=2)
    full_wavelengths = vnir_wavelengths + swir_wavelengths
    print(f"Combined HSI cube shape: {full_hsi_cube.shape}")

    # --- 2. Determine EnMap Band Indices ---
    print("Step 2: Determining EnMap band indices...")
    enmap_band_centers = {
        "blue": find_closest_band(full_wavelengths, 450),
        "green": find_closest_band(full_wavelengths, 550),
        "red": find_closest_band(full_wavelengths, 660),
        "nir": find_closest_band(full_wavelengths, 865),
    }
    print(f"Using band indices for land cover: {enmap_band_centers}")

    swir_bands_for_model_indices = np.arange(full_hsi_cube.shape[2])[-90:]
    print(f"Using the last 90 bands for model input (indices {swir_bands_for_model_indices[0]} to {swir_bands_for_model_indices[-1]})")

    # --- 3. Setup and Pre-computation ---
    print("Step 3: Setting up directories and loading signatures...")
    main_output_path = Path(args.output_dir)
    tile_inputs_path = main_output_path / "tile_inputs"
    tile_results_path = main_output_path / "tile_results"
    tile_inputs_path.mkdir(parents=True, exist_ok=True)
    tile_results_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.methane_signature_path):
        print(f"ERROR: Methane signature not found at {args.methane_signature_path}")
        return
    
    # Load the original methane signature from text file
    original_methane_signature = np.loadtxt(args.methane_signature_path)
    if original_methane_signature.ndim == 2:
        # If it's a 2D array, take the last column (which contains the signature values)
        original_methane_signature = original_methane_signature[:, -1]
    
    # Extract methane signature features for EnMAP SWIR
    # We need to sample from the original signature to match our combined band count
    start_idx = 106
    end_idx = 414
    num_combined_bands = full_hsi_cube.shape[2]  # Total VNIR + SWIR bands
    
    # Create evenly spaced indices from the range 106-414 to match our combined bands
    indices = np.linspace(start_idx, end_idx, num_combined_bands, dtype=int)
    
    # Extract the methane signature features at these indices
    methane_signature = original_methane_signature[indices]
    print(f"Resampled methane signature shape: {methane_signature.shape}")
    print(f"Original signature range: indices {start_idx}-{end_idx}, resampled to {num_combined_bands} bands")

    # --- 4. Tiling and Inference Loop ---
    print("\n--- Step 4: Starting Tiling and Inference Loop ---")
    tile_count = 0
    for tile_full_spectrum, (r, c) in tile_generator(full_hsi_cube, args.tile_size, args.overlap):
        tile_count += 1
        tile_name = f"tile_r{r}_c{c}"
        print(f"\nProcessing {tile_name}...")

        tile_input_dir = tile_inputs_path / tile_name
        tile_output_dir = tile_results_path / tile_name
        tile_input_dir.mkdir(exist_ok=True)

        print("  - Generating RGB and land cover segmentation...")
        invalid_pixel_mask = np.isnan(tile_full_spectrum).any(axis=2)
        index_generator = Indexes(train_gen_args, band_centers=enmap_band_centers)
        _, seg_map, rgb_image = index_generator.getLandCls(tile_full_spectrum, invalid_pixel_mask)

        print("  - Generating Matched Filter output...")
        mf_output = segmentation_match_filter(
            b_img_data=tile_full_spectrum,
            segmentation=seg_map,
            target=methane_signature,
            pxl_batch=train_gen_args.pxl_batch_size,
            segmentation_mf_method=train_gen_args.segmentation_mf
        )

        hsi_90_band_tile = tile_full_spectrum[:, :, swir_bands_for_model_indices]

        rgb_path = tile_input_dir / "rgb.npy"
        mf_path = tile_input_dir / "mf.npy"
        hsi_path = tile_input_dir / "hsi.npy"
        np.save(rgb_path, rgb_image)
        np.save(mf_path, mf_output)
        np.save(hsi_path, hsi_90_band_tile)

        print(f"  - Running inference on {tile_name}...")
        command = [
            "python", "inference.py",
            "--hsi_path", str(hsi_path),
            "--mf_path", str(mf_path),
            "--rgb_path", str(rgb_path),
            "--stats_file_path", args.stats_file_path,
            "--model_checkpoint", args.model_checkpoint,
            "--output_dir", str(tile_output_dir)
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"  - Inference successful. Results saved in {tile_output_dir}")
            if result.stdout:
                print("    Inference script output:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"  - ERROR during inference on {tile_name}:")
            print(e.stderr)

    print(f"\n--- Processing Complete. Processed {tile_count} tiles. ---")
    print(f"All inputs and results are stored in: {args.output_dir}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)