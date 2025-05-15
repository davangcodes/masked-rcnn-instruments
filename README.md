# 🩺 Surgical Instrument Segmentation using Mask R-CNN (CholecInstanceSeg)

This project demonstrates the training and evaluation of a **Mask R-CNN** model on the **CholecInstanceSeg** dataset using **Detectron2** for segmenting surgical instruments in laparoscopic surgery frames.

---

## 📂 Directory Structure

```
├── videos/                 # Image frames from CholecT50 dataset
│   └── VIDXX/000001.png    # Format per video and frame
├── annotations/
│   ├── train_coco.json     # All annotations converted to COCO format
│   ├── train_split.json    # Training split (90%)
│   ├── test_split.json     # Testing split (10%)
├── convert_to_coco.py      # Converts LabelMe annotations to COCO format
├── split_coco.py           # Splits train/test from COCO JSON
├── train_maskrcnn.py       # Train the Detectron2 Mask R-CNN model
├── evaluate_maskrcnn.py    # Evaluate the trained model
├── output_maskrcnn/        # Stores trained model checkpoints and logs
```

---

## ⚙️ Requirements

* Python ≥ 3.8
* `opencv-python`, `tqdm`
* [Detectron2](https://github.com/facebookresearch/detectron2)

```bash
pip install opencv-python tqdm
# Follow official instructions to install Detectron2:
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

---

## 🧾 Step-by-Step Instructions

### 1️⃣ Convert LabelMe to COCO format

```bash
python convert_to_coco.py
```

Converts all `*_full/ann_dir/*.json` files from the CholecInstanceSeg annotations into a single `train_coco.json` in COCO format. Reads corresponding images from `videos/VIDxx/`.

### 2️⃣ Split COCO Annotations into Train and Test

```bash
python split_coco.py
```

Splits the full `train_coco.json` into:

* `train_split.json` → 90%
* `test_split.json` → 10%

### 3️⃣ Train the Mask R-CNN Model

```bash
python train_maskrcnn.py
```

**Configuration:**

* Model: `mask_rcnn_R_50_FPN_3x`
* Classes: 7 (surgical instruments)
* Batch Size: 4
* Learning Rate: 0.00025
* Iterations: 5000
* Output Directory: `output_maskrcnn/`

### 4️⃣ Evaluate on Test Set

```bash
python evaluate_maskrcnn.py
```

* Uses `test_split.json`
* Computes COCO metrics:

  * `bbox mAP`
  * `segm mAP`
  * `AP50`, `AP75`, etc.

---

## 📊 Evaluation Results

### Bounding Box (bbox) AP:

| Metric              | Value |
| ------------------- | ----- |
| AP@\[IoU=0.50:0.95] | 50.2% |
| AP@\[IoU=0.50]      | 61.4% |
| AP@\[IoU=0.75]      | 58.2% |
| AP@\[small]         | 10.7% |
| AP@\[medium]        | 31.3% |
| AP@\[large]         | 52.6% |

### Segmentation (segm) AP:

| Metric              | Value |
| ------------------- | ----- |
| AP@\[IoU=0.50:0.95] | 47.3% |
| AP@\[IoU=0.50]      | 60.8% |
| AP@\[IoU=0.75]      | 54.5% |
| AP@\[small]         | 5.4%  |
| AP@\[medium]        | 25.2% |
| AP@\[large]         | 50.8% |

### Average Recall (AR):

| Metric          | BBox AR | Segm AR |
| --------------- | ------- | ------- |
| AR\@maxDets=1   | 50.7%   | 48.7%   |
| AR\@maxDets=10  | 54.7%   | 52.3%   |
| AR\@maxDets=100 | 54.7%   | 52.3%   |
| AR@\[small]     | 16.0%   | 15.1%   |
| AR@\[medium]    | 37.0%   | 33.9%   |
| AR@\[large]     | 57.0%   | 54.6%   |

---

## 🧠 Notes

* Dataset: CholecInstanceSeg
* Framework: Detectron2
* Mask R-CNN is used only to segment instruments from the CholecT50 dataset.
* Replace `NUM_CLASSES = 7` if your dataset has a different number of instrument classes.

---

## 📌 TODO

* Add instance visualization script
* Evaluate per-class mAP
* Support EndoVis 2017 dataset
* Add CLI for training config

---

## ✍️ Author

**Davang Sikand**
AI Research Fellow
Plaksha University
