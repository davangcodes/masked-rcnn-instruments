#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Trained Mask R-CNN on CholecInstanceSeg (Test Set)
-----------------------------------------------------------

This script evaluates a trained Mask R-CNN model on the test dataset 
using Detectron2's built-in COCOEvaluator.

It performs:
1. COCO-style dataset registration using `test_split.json`.
2. Loads the trained model weights from the output directory.
3. Initializes the predictor and COCO evaluator.
4. Computes detection and segmentation metrics on the test dataset.

Expected:
- Trained weights saved at `output_maskrcnn/model_final.pth`
- Evaluation metrics printed to stdout (e.g., AP50, mAP)

Author: Davang Sikand
Last updated: May 2025
"""

import os
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

# === Step 1: Register test dataset ===
dataset_name = "instrument_test"
image_root = "../../../../../../mount/Data1/Davang/CholecT50"  # Path to CholecT50 images
json_annotation = "annotations/test_split.json"  # COCO-style annotation

register_coco_instances(dataset_name, {}, json_annotation, image_root)

# === Step 2: Load model config ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))

# === Step 3: Set model weights and testing parameters ===
cfg.MODEL.WEIGHTS = os.path.join("output_maskrcnn", "model_final.pth")  # path to trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Minimum score threshold for predictions
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Number of instrument classes

# Optional: Set image resizing parameters
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333

# === Step 4: Initialize predictor and evaluator ===
predictor = DefaultPredictor(cfg)

# Build evaluation loader and evaluator
evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, dataset_name)

# === Step 5: Run inference and evaluation ===
print("🔍 Running inference on test set...")
inference_on_dataset(predictor.model, val_loader, evaluator)
