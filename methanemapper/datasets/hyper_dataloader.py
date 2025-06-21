# Copyright (c) UCSB

import json
from pathlib import Path

import numpy as np
import torch

from skimage import draw
import datasets.transforms as T
import torch.utils.data as tdata
import glob


class HyperSegment(tdata.Dataset):
    def __init__(self, img_folder, ann_file, stats_file, return_masks):
        self.data_items = LoadItems(img_folder, ann_file, stats_file)
        self._transform_rgb = makeHyperTransform(img_type="rgb")
        self._transform_mf = makeHyperTransform(img_type="mf")
        self._transform_raw = makeHyperTransform(img_type="raw", stats=self.data_items)
        self.prepare = ConvertHyperToMask(return_masks)
        self.target_keys = list(self.data_items.anns.keys())

    def __getitem__(self, idx):
        rgb_path = self.data_items.rgb_paths[idx]
        mf_path = self.data_items.mf_paths[idx]
        raw_path = self.data_items.raw_paths[idx]
        target = self.data_items.anns[self.target_keys[idx]]
        rgb_img, mf_img, raw_img, target = self.prepare(rgb_path, mf_path, raw_path, target)

        if self._transform_rgb is not None:
            rgb_img, target = self._transform_rgb(rgb_img, target)
            mf_img, _ = self._transform_mf(mf_img)
            raw_img, _ = self._transform_raw(raw_img)

        return {"rgb": rgb_img, "mf": mf_img, "raw": raw_img, "target": target}

    def __len__(self):
        return len(self.data_items.rgb_paths)


def makeHyperTransform(img_type, stats=None):
    if img_type == "rgb":
        normalize = T.Compose(
            [
                T.ScaleRGB(),
                T.ToTensorHyper(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # T.RandomHorizontalFlip()
            ]
        )
        return T.Compose([normalize])

    if img_type == "mf":
        normalize = T.Compose(
            [
                T.ToTensorHyper(),
                T.Normalize([0.49631513], [0.08298772]),
                # T.RandomHorizontalFlip()
            ]
        )
        return T.Compose([normalize])

    if img_type == "raw":
        normalize = T.Compose(
            [
                T.RemoveMaskedArea(mask_val=-49.0),
                T.ToTensorHyper(),
                T.Normalize(stats.mean[-90:].tolist(), stats.std[-90:].tolist()),
                # T.RandomHorizontalFlip()
            ]
        )
        return T.Compose([normalize])

    """
	if image_set == 'train':
		return T.Compose([
			normalize,
		])

	if image_set == 'val':
		return T.Compose([
			normalize,
		])

	raise ValueError(f'unknown {image_set}')
	"""


class LoadItems:
    def __init__(self, img_folder, ann_file, stats_file):
        # load dataset
        self.anns, self.imgs = dict(), dict()
        self.img_dir = img_folder # methanemapper/data/train
        self.mean = None
        self.std = None

        if not ann_file == None:
            print("loading annotations into memory...")
            try:
                dataset = json.load(open(ann_file, "r"))
                assert type(dataset) == dict, "annotation file format {} not supported".format(type(dataset))
                self.dataset = dataset
                self.createList()
                self.createPaths()
            except Exception as e:
                print(f"Warning: Error loading annotations: {e}")
                # Initialize empty data structures as fallback
                self.anns = {}
                self.img_id = {}
                self.patch_id = {}
                self.rgb_paths = []
                self.mf_paths = []
                self.raw_paths = []

        if not stats_file == None:
            print("loading mean and std for each band...")
            try:
                self.mean = np.load(f"{stats_file}/dataset_mean.npy")
                self.std = np.load(f"{stats_file}/dataset_std.npy")
            except Exception as e:
                print(f"Warning: Error loading statistics: {e}")
                # Use default values as fallback
                self.mean = np.zeros(90)
                self.std = np.ones(90)

    def createList(self):
        # create list of all images and annotations
        print("creating annotations list")
        anns, img_id, patch_id = dict(), dict(), dict()
        # FIXED : fix the missing annotations because of same file names
        unq_id = 1  # assigning a id to each image file, at train time it's the image id
        
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                try:
                    # Use get method with defaults to handle missing keys
                    patch_name = ann.get('patch_name', f"unknown_{unq_id}")
                    anns[f"{patch_name}"] = {
                        "segmentation": ann.get('segmentation', []),
                        "bbox": ann.get('bbox', [0, 0, 10, 10]),  # default small box
                        "category_id": ann.get('category_id', 1),  # default category
                        "image_id": unq_id,
                    }
                    img_id[f"{patch_name}"] = ann.get('image_id', f"img_{unq_id}")
                    patch_id[f"{patch_name}"] = ann.get('patch_id', f"patch_{unq_id}")
                    unq_id += 1
                except Exception as e:
                    print(f"Warning: Error processing annotation: {e}")
                    continue
            
            if anns:
                print(f"Successfully loaded {len(anns)} annotations")
                print("Sample annotation:", next(iter(anns.values())))
        else:
            print("Warning: No 'annotations' key found in dataset")

        # create class members
        self.anns = anns
        self.img_id = img_id
        self.patch_id = patch_id

    def createPaths(self):
        # create a list of all rgb, mf, raw images
        print("creating image paths list")
        rgb_paths, mf_paths, raw_paths = [], [], []

        # Process each annotation
        for _ann_key in list(self.anns.keys()):
            try:
                _iid = self.img_id.get(_ann_key, "unknown")
                _pid = self.patch_id.get(_ann_key, "unknown")
                
                # Find RGB file
                rgb_glob = glob.glob(f"{self.img_dir}/rgb_tiles/{_iid}_*/*_{_pid}.npy")
                if not rgb_glob:
                    # print(f"Warning: No RGB file found for {_iid}, {_pid}")
                    continue
                
                _rgb = rgb_glob[0]
                _tmp = _rgb.split("/")
                
                # Construct MF and RAW paths
                _mf = f"{self.img_dir}/mf_tiles/{_tmp[-2]}/{_tmp[-1]}"
                _raw = f"{self.img_dir}/rdata_tiles/{_tmp[-2]}/{_tmp[-1]}"
                
                # Verify file existence
                if not (Path(_mf).exists() and Path(_raw).exists()):
                    # print(f"Warning: Missing files for {_ann_key}: MF={Path(_mf).exists()}, RAW={Path(_raw).exists()}")
                    continue
                
                # Check image dimensions to ensure they are 256x256 and not corrupt
                try:
                    # rgb_shape = np.load(_rgb, mmap_mode='r').shape
                    mf_shape = np.load(_mf, mmap_mode='r').shape
                    # raw_shape = np.load(_raw, mmap_mode='r').shape
                    # print(f'rgb_shape: {rgb_shape}, mf_shape: {mf_shape}, raw_shape: {raw_shape}')

                    if not (mf_shape[0] > 200 and mf_shape[1] > 200):
                        continue

                    # if not (rgb_shape[:-1] == mf_shape and mf_shape == raw_shape[:-1]
                    #         and rgb_shape[:-1] == raw_shape[:-1]):
                    #     continue

                    # if not (
                    #         # rgb_shape[0] == 256 and rgb_shape[1] == 256 and
                    #          mf_shape[0] == 256 and mf_shape[1] == 256 
                    #         # and raw_shape[0] == 256 and raw_shape[1] == 256
                    #         ):
                    #     continue  # Skip if not 256x256
                except Exception as e:
                    # This will also catch the corrupted files
                    # print(f"Warning: Could not load or check shape for {_ann_key}: {e}")
                    continue
                
                rgb_paths.append(_rgb)
                mf_paths.append(_mf)
                raw_paths.append(_raw)
                
            except Exception as e:
                print(f"Warning: Error processing paths for {_ann_key}: {e}")
                continue

        print(f"Found {len(rgb_paths)} valid image sets out of {len(self.anns)} annotations")
        
        if len(rgb_paths) == 0:
            print("Warning: No valid image paths found. Using empty lists.")

        # create class members
        self.rgb_paths = rgb_paths
        self.mf_paths = mf_paths
        self.raw_paths = raw_paths


class ConvertHyperToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, rgb_path, mf_path, raw_path, target):
        # dim order according to coco : ch, h, w
        rgb_img = np.load(rgb_path).transpose((2, 0, 1))
        mf_img = np.expand_dims(np.load(mf_path), axis=2).transpose((2, 0, 1))
        raw_img = np.load(raw_path).transpose((2, 0, 1))

        _, h, w = rgb_img.shape

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # def _plotGTforVerify(mf_path, rgb_path, target):
        #     import cv2
        #     import pdb

        #     pdb.set_trace()
        #     # mf_img = np.load(mf_path)*255
        #     rgb_img = np.load(rgb_path)
        #     target_seg = target["segmentation"]
        #     all_segs = []
        #     for _seg in target_seg:
        #         _seg = np.array(_seg)
        #         all_segs.append(np.expand_dims(_seg, axis=1))
        #     all_segs = tuple(all_segs)
        #     # mf_img = cv2.cvtColor(np.uint8(mf_img), cv2.COLOR_GRAY2RGB)
        #     cv2.drawContours(rgb_img, all_segs, -1, (255, 0, 0), 3)
        #     cont_img_path = (
        #         f"/data/satish/REFINERY_DATA/hyper_detr/data/visual_gt/{mf_path.split('/')[-1].split('.')[0]}.png"
        #     )
        #     cv2.imwrite(cont_img_path, rgb_img)

        # _plotGTforVerify(mf_path, rgb_path, target)

        boxes = target["bbox"]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = target["category_id"]
        classes = torch.tensor(classes, dtype=torch.int64)

        masks = []
        if self.return_masks:
            segmentations = target["segmentation"]
            for _seg in segmentations:
                # convert boundary pixels to binary segmentation mask
                masks.append(draw.polygon2mask((w, h), np.array(_seg)).T)  # this function takes image shape in w,h
            masks = np.array(masks)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = torch.from_numpy(masks)
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        if self.return_masks:
            target["masks"] = masks

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return rgb_img, mf_img, raw_img, target


def buildHyperSeg(image_set, args):
    if image_set == "train":
        img_folder = Path(args.train_img_folder)
        ann_file = Path(args.train_ann_file)
    elif image_set == "val":
        img_folder = Path(args.val_img_folder)
        ann_file = Path(args.val_ann_file)
    else:
        raise ValueError(f"Unknown image_set: {image_set}")

    stats_file = Path(args.stats_file_path)

    assert img_folder.exists(), f"provided image folder path {img_folder} does not exist"
    assert ann_file.exists(), f"provided annotation file path {ann_file} does not exist"
    assert stats_file.exists(), f"provided stats file path {stats_file} does not exist"

    dataset = HyperSegment(img_folder, ann_file, stats_file, return_masks=args.masks)

    return dataset
