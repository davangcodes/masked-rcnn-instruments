#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=================================================================================
 Script Name: convert_to_coco.py
 Author: Davang Sikand
 Purpose: Convert LabelMe-style polygon annotations to COCO format
=================================================================================

📌 USE CASE:
-------------
This script is designed to convert surgical instance segmentation annotations 
(from the CholecInstanceSeg dataset in LabelMe JSON format) to COCO format, 
which is required for training models like Detectron2’s Mask R-CNN.

📁 INPUT STRUCTURE:
-------------------
- base_annotation_dir (LabelMe JSON files):
    /path/to/cholecinstanceseg/train/
        └── VID01_full/
            └── ann_dir/
                ├── t50_VID01_000001.json
                ├── t50_VID01_000002.json
                └── ...

- base_image_dir (Actual image files):
    /path/to/CholecT50/
        └── videos/
            └── VID01/
                ├── 000001.png
                ├── 000002.png
                └── ...

📤 OUTPUT:
----------
- A single COCO-style JSON file with:
    ├── images: image metadata
    ├── annotations: per-object polygon + bbox
    └── categories: list of unique labels

✅ WHAT THE SCRIPT DOES:
-------------------------
1. Reads each LabelMe JSON file
2. Extracts polygon points and labels
3. Associates them with the corresponding image
4. Computes bounding boxes and area
5. Organizes everything into COCO-compliant format
6. Saves the final dataset into a JSON file

🔐 DEPENDENCIES:
----------------
- Python 3.x
- OpenCV (cv2)
- tqdm
- glob
- json

🚀 HOW TO RUN:
--------------
python convert_to_coco.py

Or integrate it in a larger preprocessing pipeline.

=================================================================================
"""

import os
import json
import cv2
from glob import glob
from tqdm import tqdm


def labelme_to_coco(base_annotation_dir, base_image_dir, output_json_path):
    """
    Converts LabelMe-style annotations to COCO format for Detectron2/Mask R-CNN.

    Args:
        base_annotation_dir (str): Directory containing *_full/ann_dir/*.json files
        base_image_dir (str): Root folder containing actual image files
        output_json_path (str): Output file path for the COCO JSON
    """
    
    annotation_files = glob(os.path.join(base_annotation_dir, "*_full", "ann_dir", "*.json"))

    coco_images = []
    coco_annotations = []
    coco_categories = []

    category_name_to_id = {}
    category_id_counter = 1
    annotation_id = 1
    image_id_map = {}
    image_id_counter = 1

    for json_path in tqdm(annotation_files, desc="Converting"):
        with open(json_path, 'r') as f:
            annotation = json.load(f)

        raw_frame = os.path.basename(json_path).split(".")[0]  # e.g., t50_VID01_000468
        frame_id = raw_frame.split("_")[-1]
        video_id = raw_frame.split("_")[1]

        filename_png = f"{frame_id}.png"
        filename_jpg = f"{frame_id}.jpg"
        relative_path_png = os.path.join("videos", video_id, filename_png)
        relative_path_jpg = os.path.join("videos", video_id, filename_jpg)
        full_path_png = os.path.join(base_image_dir, relative_path_png)
        full_path_jpg = os.path.join(base_image_dir, relative_path_jpg)

        if os.path.exists(full_path_png):
            image_path = full_path_png
            relative_path = relative_path_png
        elif os.path.exists(full_path_jpg):
            image_path = full_path_jpg
            relative_path = relative_path_jpg
        else:
            print(f"⚠️ Image not found for frame {raw_frame} in {video_id}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read image: {image_path}")
            continue
        height, width = img.shape[:2]

        if relative_path not in image_id_map:
            image_id = image_id_counter
            image_id_map[relative_path] = image_id
            coco_images.append({
                "id": image_id,
                "file_name": relative_path,
                "height": height,
                "width": width
            })
            image_id_counter += 1
        else:
            image_id = image_id_map[relative_path]

        for shape in annotation.get("shapes", []):
            label = shape["label"]
            points = shape["points"]

            if label not in category_name_to_id:
                category_name_to_id[label] = category_id_counter
                category_id_counter += 1

            category_id = category_name_to_id[label]

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min = min(x_coords)
            y_min = min(y_coords)
            bbox_width = max(x_coords) - x_min
            bbox_height = max(y_coords) - y_min

            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [sum(points, [])],
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            annotation_id += 1

    coco_categories = [
        {"id": cid, "name": name}
        for name, cid in category_name_to_id.items()
    ]

    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f)

    print(f"\n✅ COCO JSON saved at: {output_json_path}")
    print(f"🖼️  Total images: {len(coco_images)}")
    print(f"🔖 Total annotations: {len(coco_annotations)}")
    print(f"🏷️  Categories: {list(category_name_to_id.keys())}")


# === Script entrypoint ===
if __name__ == "__main__":
    # Modify these paths based on your setup
    base_annotation_dir = "../../../../../../mount/Data1/Davang/cholecinstanceseg/train"
    base_image_dir = "../../../../../../mount/Data1/Davang/CholecT50"
    output_json_path = "annotations/train_coco.json"

    labelme_to_coco(base_annotation_dir, base_image_dir, output_json_path)
