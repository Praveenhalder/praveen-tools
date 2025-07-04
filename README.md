# Praveen Tools for ComfyUI

A collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed to simplify and enhance image processing workflows. These tools offer utility functions for image selection, splitting/merging batches, post-processing enhancements, and dimension validation tailored for models like **Stable Diffusion**.

---

## 🧩 Node Features

### 📌 SelectLastImage  
**Category:** `image/utility`  
**Description:** Selects the last image from a batch of images.  
**Input:** Optional list of images  
**Output:** Single image

---

### 📌 SplitImageList  
**Category:** `image/list`  
**Description:** Splits a batch of images into three separate lists based on customizable ratios.  
**Inputs:**  
- `images` (IMAGE)  
- `split_ratio_1`, `split_ratio_2` (FLOAT)  
**Outputs:**  
- Three separate image batches

---

### 📌 MergeImageLists  
**Category:** `image/list`  
**Description:** Merges three image batches back into a single list, with customizable order.  
**Inputs:**  
- `images_1`, `images_2`, `images_3` (IMAGE)  
- `merge_order`: Order in which to merge (e.g., `"1-2-3"`)  
**Output:** Combined image batch

---

### 📌 AdjustBrightnessContrast  
**Category:** `image/postprocessing`  
**Description:** Adjust brightness, contrast, saturation, and individual RGB gains for an image.  
**Inputs:**  
- `brightness`, `contrast`, `saturation` (FLOAT)  
- `red_gain`, `green_gain`, `blue_gain` (FLOAT)  
**Output:** Enhanced image

---

### 📌 ImageDimensions16  
**Category:** `image/dimensions`  
**Description:** Ensures image width and height are multiples of 16 and length is a multiple of 4 plus 1 (as required by some models).  
**Inputs:**  
- `width`, `height`, `length`  
**Outputs:** Validated/resized dimensions

---

## 🔧 Installation

1. Copy `Praveen_tools.py` into your ComfyUI custom nodes folder:
   ```bash
   ComfyUI/custom_nodes/praveen_tools/Praveen_tools.py
