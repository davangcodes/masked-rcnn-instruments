#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=================================================================================
 Script Name: split_coco_train_test.py
 Author: Davang 
 Purpose: Split a single COCO-format JSON file into training and testing splits
=================================================================================

📌 USE CASE:
-------------
Once you have a full `train_coco.json` file (converted from LabelMe using 
`convert_to_coco.py`), this script helps you split that dataset into 
training and testing sets in COCO format. This is useful for Mask R-CNN 
training and evaluation using Detectron2 or similar frameworks.

📁 INPUT:
---------
- in_path: COCO JSON file path (e.g., "annotations/train_coco.json")

📤 OUTPUT:
----------
- annotations/train_split.json : subset of images + annotations for training
- annotations/test_split.json  : subset of images + annotations for testing

🧮 PARAMETERS:
--------------
- test_ratio: ratio of images to reserve for testing (default: 10%)
- seed: random seed for reproducibility

🚀 HOW TO RUN:
--------------
python split_coco_train_test.py

=================================================================================
"""

import json
import os
import random


def split_coco_train_test(
    in_path="annotations/train_coco.json",
    out_dir="annotations",
    test_ratio=0.1,
    seed=42
):
    # Load the input COCO-style dataset
    with open(in_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Shuffle image IDs reproducibly
    random.seed(seed)
    img_ids = [img["id"] for img in images]
    random.shuffle(img_ids)

    # Calculate split sizes
    n_test = int(len(img_ids) * test_ratio)
    test_ids = set(img_ids[:n_test])
    train_ids = set(img_ids[n_test:])

    # Split images and annotations
    train_images = [img for img in images if img["id"] in train_ids]
    test_images  = [img for img in images if img["id"] in test_ids]

    train_annots = [ann for ann in annotations if ann["image_id"] in train_ids]
    test_annots  = [ann for ann in annotations if ann["image_id"] in test_ids]

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save split JSONs
    with open(os.path.join(out_dir, "train_split.json"), "w") as f:
        json.dump({"images": train_images, "annotations": train_annots, "categories": categories}, f)

    with open(os.path.join(out_dir, "test_split.json"), "w") as f:
        json.dump({"images": test_images, "annotations": test_annots, "categories": categories}, f)

    # Summary
    print(f"✅ Wrote {len(train_images)} train images → annotations/train_split.json")
    print(f"✅ Wrote {len(test_images)}  test images  → annotations/test_split.json")


# Entrypoint
if __name__ == "__main__":
    split_coco_train_test(
        in_path="annotations/train_coco.json",
        out_dir="annotations",
        test_ratio=0.1,
        seed=42
    )
