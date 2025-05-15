
markdown
Copy
Edit
# 🩺 Surgical Instrument Segmentation using Mask R-CNN (CholecInstanceSeg)

This project demonstrates the training and evaluation of a **Mask R-CNN** model on the **CholecInstanceSeg** dataset using **Detectron2** for segmenting surgical instruments in laparoscopic surgery frames.

---

## 📂 Directory Structure

├── videos/ # Image frames from CholecT50 dataset
│ └── VIDXX/000001.png # Format per video and frame
├── annotations/
│ ├── train_coco.json # All annotations converted to COCO format
│ ├── train_split.json # Training split (90%)
│ ├── test_split.json # Testing split (10%)
├── convert_to_coco.py # Converts LabelMe annotations to COCO format
├── split_coco.py # Splits train/test from COCO JSON
├── train_maskrcnn.py # Train the Detectron2 Mask R-CNN model
├── evaluate_maskrcnn.py # Evaluate the trained model
├── output_maskrcnn/ # Stores trained model checkpoints and logs

yaml
Copy
Edit

---

## ⚙️ Requirements

- Python ≥ 3.8
- `opencv-python`, `tqdm`
- [Detectron2](https://github.com/facebookresearch/detectron2)

```bash
pip install opencv-python tqdm
# Follow official instructions to install Detectron2:
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
🧾 Step-by-Step Instructions
1️⃣ Convert LabelMe to COCO format
bash
Copy
Edit
python convert_to_coco.py
Converts all *_full/ann_dir/*.json files from the CholecInstanceSeg annotations into a single train_coco.json in COCO format.

Reads corresponding images from videos/VIDxx/.

2️⃣ Split COCO Annotations into Train and Test
bash
Copy
Edit
python split_coco.py
Splits the full train_coco.json into:

train_split.json → 90%

test_split.json → 10%

3️⃣ Train the Mask R-CNN Model
bash
Copy
Edit
python train_maskrcnn.py
Configuration:

Model: mask_rcnn_R_50_FPN_3x

Classes: 7 (surgical instruments)

Batch Size: 4

Learning Rate: 0.00025

Iterations: 5000

Output Directory: output_maskrcnn/

4️⃣ Evaluate on Test Set
bash
Copy
Edit
python evaluate_maskrcnn.py
Uses test_split.json

Computes COCO metrics:

bbox mAP

segm mAP

AP50, AP75, etc.

📊 Example Results
Metric	Value (example)
mAP (IoU=0.50)	82.4%
mAP (IoU=0.75)	69.3%
Overall mAP	65.1%

🧠 Notes
Dataset: CholecInstanceSeg

Framework: Detectron2

Mask R-CNN is only used to segment instruments from the CholecT50 dataset.

Replace NUM_CLASSES = 7 if your dataset uses a different number of instrument classes.

📌 TODO
 Add instance visualization script

 Evaluate per-class mAP

 Support EndoVis 2017 dataset

 Add CLI for training config

✍️ Author
Davang Sikand
AI Research Fellow
Plaksha University

yaml
Copy
Edit

---

Let me know if you want a version tailored for GitHub Pages or if you'd like me to auto-generate badges (license, Python version, etc.) for the top of the README.






