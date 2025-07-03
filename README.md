# 🧩 ComfyUI - Image List Utility Nodes

A set of custom nodes for **ComfyUI** that allows advanced manipulation of image lists: splitting, merging, and selecting images efficiently within workflows.

## 📦 Nodes Included

### 1. **SelectLastImage**
Selects the **last image** from a batch or list of images.

- **Inputs:**
  - `images` (IMAGE): Optional, force input required.
- **Outputs:**
  - `image` (IMAGE): The last image from the list.

### 2. **SplitImageList**
Splits a list of images into **three parts** based on configurable ratios.

- **Inputs:**
  - `images` (IMAGE): Required image list.
  - `split_ratio_1` (FLOAT): Ratio for the first split (default: 0.33).
  - `split_ratio_2` (FLOAT): Ratio for the second split (default: 0.33).
- **Outputs:**
  - `images_1`, `images_2`, `images_3`: Tensors representing the three image segments.

### 3. **MergeImageLists**
Merges **three image lists** into one combined list, preserving or customizing the order.

- **Inputs:**
  - `images_1`, `images_2`, `images_3` (IMAGE): Image tensors or lists.
  - `merge_order` (OPTIONAL): Choose merge order (default: "1-2-3").
- **Outputs:**
  - `merged_images` (IMAGE): A single merged image list.

---

## 🛠️ Installation

1. Place the Python file (e.g., `image_list_utils.py`) into your `ComfyUI/custom_nodes/` folder.
2. Restart **ComfyUI**.
3. Search in the node editor:
   - `"Select Last Image"`
   - `"Split Image List (3-Way)"`
   - `"Merge Image Lists (3-Way)"`

---

## 🧪 Example Use Case

### 🔄 Reorder a List of Generated Images

1. Generate a batch of images.
2. Use **SplitImageList** to divide them into sections (e.g., intro, mid, outro).
3. Reorder sections using **MergeImageLists** with a custom order like `"2-3-1"`.
4. Use **SelectLastImage** if only the final frame is needed for output.

---

## 📋 Notes

- Handles both single image tensors and image lists.
- Safeguards against empty or improperly shaped inputs.
- Split ratios are automatically normalized and validated to avoid out-of-range slicing.

---

## 🔧 Requirements

- **ComfyUI** installed
- Compatible with **PyTorch** image tensors used in ComfyUI workflows

---

## 📄 License

This project is open-source under the MIT License.

---

## ✨ Credits

Developed by Praveen


