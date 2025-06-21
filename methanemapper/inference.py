# Copyright (c) UCSB
#
# This script is designed for running inference with a trained MethaneMapper model
# on a new Hyperspectral Imaging (HSI) cube. It demonstrates the pipeline needed
# to go from a raw HSI cube to methane detection predictions (bounding boxes/masks).
#
# As the preprocessing steps (e.g., creating RGB images and Matched Filter outputs)
# can be specific to the sensor and data format, this script provides a template
# and requires the user to adapt the data loading and preprocessing parts to their needs.

import argparse
import json
import cv2
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# --- Project Imports ---
from models import build_model
from datasets.hyper_dataloader import makeHyperTransform, LoadItems
from util.misc import NestedTensor


# The Matched Filter (MF) computation is critical. The original project has a script
# for this in `train_dataset_generator/train_data_generator.py`. You will need to
# implement a similar function here or preprocess your data to generate the MF input.
# from spectral import matched_filter

def get_args_parser():
    """
    Defines the command-line arguments for the inference script.
    """
    parser = argparse.ArgumentParser("MethaneMapper Inference Script")

    # --- Inference-specific Arguments ---
    parser.add_argument(
        "--hsi_path",
        type=str,
        required=True,
        help="Path to the raw HSI data cube (e.g., a .npy file). Expected shape: [height, width, bands]. The bands should match the 90 bands used in training.",
    )
    parser.add_argument(
        "--mf_path",
        type=str,
        required=True,
        help="Path to the Matched Filter (MF) output for the HSI cube (e.g., a .npy file). Expected shape: [height, width].",
    )
    parser.add_argument(
        "--rgb_path",
        type=str,
        required=True,
        help="Path to the RGB representation of the HSI cube (e.g., a .npy or image file).",
    )
    parser.add_argument(
        "--stats_file_path",
        type=str,
        required=True,
        help="Path to the directory with statistics files (dataset_mean.npy, dataset_std.npy).",
    )
    parser.add_argument(
        "--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--output_dir", default="./inference_results", help="Path to save inference results (predictions.json and visualization.png)."
    )
    parser.add_argument("--device", default="cuda", help="Device to use for inference ('cuda' or 'cpu').")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold to filter out low-confidence predictions."
    )

    return parser


def preprocess_input(hsi_cube, mf_output, rgb_image, stats_file_path, device):
    """
    Preprocesses the input data by applying necessary transformations and normalization.
    """
    data_items = LoadItems(img_folder=None, ann_file=None, stats_file=stats_file_path)

    # --- Define Transformations ---
    # These should match the validation transforms used during training
    transform_rgb = makeHyperTransform('rgb')
    transform_mf = makeHyperTransform('mf')
    transform_raw = makeHyperTransform('raw', stats=data_items)

    # --- Apply Transformations ---
    # The transforms expect numpy arrays with shape (C, H, W)
    rgb_tensor, _ = transform_rgb(rgb_image.transpose((2, 0, 1)), None)
    mf_tensor, _ = transform_mf(mf_output.transpose((2, 0, 1)), None)
    raw_tensor, _ = transform_raw(hsi_cube.transpose((2, 0, 1)), None)

    # Add batch dimension and move to device
    return (
        rgb_tensor.unsqueeze(0).to(device),
        mf_tensor.unsqueeze(0).to(device),
        raw_tensor.unsqueeze(0).to(device),
    )


def main(inference_args):
    """
    Main function to run the inference pipeline.
    """
    print("--- MethaneMapper Inference ---")

    # 1. Load checkpoint and training arguments
    print(f"Loading model from checkpoint: {inference_args.model_checkpoint}")
    checkpoint = torch.load(inference_args.model_checkpoint, map_location="cpu")
    
    if "args" not in checkpoint:
        print("ERROR: The model checkpoint does not contain training arguments ('args').")
        print("Please use a checkpoint saved from the main training script or manually provide all model architecture arguments.")
        return

    train_args = checkpoint["args"]

    # 2. Update training args with inference-specific ones
    # This allows overriding settings like device, output_dir, etc.
    vars(train_args).update(vars(inference_args))
    args = train_args
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load Input Data
    print(f"Loading HSI data from: {args.hsi_path}")
    hsi_cube = np.load(args.hsi_path).astype(np.float32)
    print(f"Loading Matched Filter data from: {args.mf_path}")
    mf_output = np.load(args.mf_path).astype(np.float32)
    print(f"Loading RGB image from: {args.rgb_path}")
    
    if args.rgb_path.endswith(".npy"):
        rgb_image = np.load(args.rgb_path)
    else:
        rgb_image = np.array(Image.open(args.rgb_path).convert("RGB"))

    # Ensure inputs have correct dimensions
    if len(mf_output.shape) == 2:
        mf_output = np.expand_dims(mf_output, axis=-1)  # Add channel dim: [H, W, 1]
    assert hsi_cube.shape[:-1] == mf_output.shape[:-1] == rgb_image.shape[:-1], "Input dimensions do not match!"
    assert hsi_cube.shape[2] == 90, f"Expected 90 HSI bands, but got {hsi_cube.shape[2]}"

    # 4. Preprocess Data
    print("Preprocessing input data...")
    rgb_tensor, mf_tensor, raw_tensor = preprocess_input(
        hsi_cube, mf_output, rgb_image, args.stats_file_path, device
    )

    # 5. Load Model
    print("Building model with loaded training arguments...")
    # The model is built using the arguments it was trained with.
    model, _, postprocessors = build_model(args)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    # 6. Run Inference
    print("Running inference on the model...")
    with torch.no_grad():
        mask = torch.zeros((1, hsi_cube.shape[0], hsi_cube.shape[1]), dtype=torch.bool, device=device)
        samples = NestedTensor(rgb_tensor, mask)
        samples1 = NestedTensor(mf_tensor, mask)
        samples2 = NestedTensor(raw_tensor, mask)
        outputs = model(samples, samples1, samples2)

    # 7. Post-process Output
    print("Post-processing results...")
    orig_size = torch.as_tensor([hsi_cube.shape[0], hsi_cube.shape[1]]).unsqueeze(0).to(device)

    postprocessor_key = "segm" if args.masks else "bbox"
    if postprocessor_key not in postprocessors:
        print(f"ERROR: Postprocessor '{postprocessor_key}' not found. Check if the '--masks' flag is set correctly based on the trained model.")
        return
        
    results = postprocessors[postprocessor_key](outputs, orig_size)

    scores = results[0]["scores"]
    keep = scores > args.threshold

    final_boxes = results[0]["boxes"][keep].cpu().numpy()
    final_scores = scores[keep].cpu().numpy()
    final_labels = results[0]["labels"][keep].cpu().numpy()

    final_masks = None
    if args.masks and "masks" in results[0]:
        final_masks = results[0]["masks"][keep].cpu().numpy()

    # 8. Save Results
    print(f"Found {len(final_scores)} objects with confidence > {args.threshold}")

    output_json_path = output_dir / "predictions.json"
    predictions = {
        "boxes": [box.tolist() for box in final_boxes],
        "scores": final_scores.tolist(),
        "labels": final_labels.tolist(),
    }
    with open(output_json_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_json_path}")

    vis_img = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Plume: {final_scores[i]:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if final_masks is not None:
        for mask in final_masks:
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
            overlay = vis_img.copy()
            overlay[mask.squeeze() > 0.5] = color
            vis_img = cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0)

    output_image_path = output_dir / "visualization.png"
    cv2.imwrite(str(output_image_path), vis_img)
    print(f"Visualization saved to {output_image_path}")
    print("--- Inference complete ---")


if __name__ == "__main__":
    parser = get_args_parser()
    inference_args = parser.parse_args()
    main(inference_args)