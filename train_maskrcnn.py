#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Mask R-CNN on CholecInstanceSeg Dataset using Detectron2
---------------------------------------------------------------

This script trains a Mask R-CNN model using the Detectron2 framework
on instrument segmentation annotations in COCO format.

Steps Covered:
1. Registers the COCO-style training and test datasets.
2. Loads a base Mask R-CNN config (R_50 + FPN).
3. Updates training hyperparameters like learning rate, batch size, iterations.
4. Sets number of instrument classes (7 in CholecInstanceSeg).
5. Initializes Detectron2's `DefaultTrainer` and starts training.
6. Saves logs and weights to `output_maskrcnn/`.

Requirements:
- COCO-style JSON annotations (`train_split.json`, `test_split.json`)
- Folder structure: CholecT50/images/VIDxx/000001.png
- Detectron2 installed correctly.

Author: Davang Sikand
Last updated: May 2025
"""

import os
import logging
from detectron2.utils.logger import setup_logger

# Initialize logging for Detectron2
setup_logger()

# === Detectron2 core components ===
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

# === Define Paths ===
DATASET_ROOT = "../../../../../../mount/Data1/Davang/CholecT50"
TRAIN_JSON = "annotations/train_split.json"
TEST_JSON = "annotations/test_split.json"
OUTPUT_DIR = "./output_maskrcnn"

# === Register Datasets ===
# Register COCO-style train/test datasets for Detectron2
register_coco_instances("cholec_train", {}, TRAIN_JSON, DATASET_ROOT)
register_coco_instances("cholec_test", {}, TEST_JSON, DATASET_ROOT)

# === Configure the model ===
cfg = get_cfg()

# Load base config for Mask R-CNN (ResNet-50 + FPN)
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))

# Dataset assignment
cfg.DATASETS.TRAIN = ("cholec_train",)
cfg.DATASETS.TEST = ("cholec_test",)
cfg.DATALOADER.NUM_WORKERS = 4

# Load pre-trained COCO weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

# Training hyperparameters
cfg.SOLVER.IMS_PER_BATCH = 4            # Number of images per batch
cfg.SOLVER.BASE_LR = 0.00025            # Learning rate
cfg.SOLVER.MAX_ITER = 5000              # Total number of iterations

# ROI Head configuration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7     # Number of instrument categories in CholecInstanceSeg

# Output directory setup
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# === Launch training ===
if __name__ == "__main__":
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
