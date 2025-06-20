import os
import json
import cv2
import numpy as np
import argparse

info = {"year": 2022, "version": 1.0}

license = [{"id": 0, "name": "Unknown", "url": ""}]
categories = [
    {"supercategory": "type", "id": 1, "name": "point_source"},
    {"supercategory": "type", "id": 2, "name": "diffused_source"},
    {"supercategory": "type", "id": 3, "name": "Unknown"},
]


def getMask(_dir_path, all_files, json_annot):
    for _file in all_files:
        file_path = os.path.join(_dir_path, _file)
        if _file.split(".")[-1] != "npy":
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        else:
            img = np.load(file_path)

        if len(img.shape) > 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        mask = np.zeros((img_gray.shape))
        mask[img_gray > 0] = 255
        # find all the independent plumes
        plume_mask = []
        plume_bbox = []
        plume_cat = []

        _temp_patch = _file.split(".")[0]
        patch_name = f"{_temp_patch}"
        _temp_patch = _temp_patch.split("_")
        patch_id = f"{_temp_patch[-2]}_{_temp_patch[-1]}"
        file_name = _file[:18]
        contours, hier = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            for _contour in contours:
                if len(_contour) < 8:
                    continue  # plume too small
                sqz_contour = _contour.squeeze(axis=1)

                x_min = int(sqz_contour[:, 0].min())
                y_min = int(sqz_contour[:, 1].min())
                x_max = int(sqz_contour[:, 0].max())
                y_max = int(sqz_contour[:, 1].max())

                # Bbox format is [x_min, y_min, width, height]
                plume_bbox.append([x_min, y_min, x_max - x_min, y_max - y_min])
                plume_mask.append(sqz_contour.tolist())
                plume_cat.append(1)

            if plume_mask:
                annot_template = {
                    "patch_name": patch_name,
                    "patch_id": patch_id,
                    "image_id": file_name,
                    "segmentation": plume_mask,
                    "bbox": plume_bbox,
                    "category_id": plume_cat,
                }
                json_annot.append(annot_template)


def main(input_dir, dest_filepath):
    json_annot = []
    all_dirs = os.listdir(input_dir)
    all_dir_path = [os.path.join(input_dir, _dir_name) for _dir_name in all_dirs]
    for _dir_path in all_dir_path:
        if not os.path.isdir(_dir_path):
            continue
        sub_dir_files = os.listdir(_dir_path)
        print("Computing annotations for directory:", _dir_path)
        getMask(_dir_path, sub_dir_files, json_annot)

    # create json data
    json_data = {"info": info, "licenses": license, "annotations": json_annot, "categories": categories}
    # write data to json file
    if os.path.isfile(dest_filepath):
        os.remove(dest_filepath)
    with open(dest_filepath, mode="w") as f:
        f.write(json.dumps(json_data, indent=4))
    print(f"\nSuccessfully created annotation file at: {dest_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COCO-style annotations from mask image tiles.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../data/all_gt_masks",
        help="The directory containing subdirectories of mask tiles.",
    )
    parser.add_argument(
        "--dest_filepath",
        type=str,
        default="./annotations.json",
        help="The file path for the output annotation JSON file.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
    else:
        main(args.input_dir, args.dest_filepath)
