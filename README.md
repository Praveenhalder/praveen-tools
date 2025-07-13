# Praveen Tools for ComfyUI


This repository provides a collection of **custom ComfyUI nodes** designed for advanced image list manipulation, image property adjustment, and workflow automation. These nodes enhance ComfyUI's capabilities by enabling batch operations, RGB merging, image generation, and dimension handling.

---

## 📦 Included Nodes

### 🖼️ Image Manipulation

- **`SelectLastImage`**  
  Select the last image from a batch.

- **`SplitImageList`**  
  Split an image list into three parts using adjustable ratios.

- **`MergeImageLists`**  
  Merge three separate image lists back into one, with configurable order.

- **`SkipFirstImage`**  
  Skips the first image from a batch—useful in looped workflows.

- **`BlackImageGenerator`**  
  Generate a black image with configurable width, height, and batch size.

- **`AdjustBrightnessContrast`**  
  Adjust brightness, contrast, saturation, and RGB gains per channel.

### 📏 Resolution Utilities

- **`ImageDimensions16`**  
  Return fixed dimensions (width, height, length) in multiples of 16.

- **`ResolutionNode`**  
  Return width and height as integers (multiple of 16), useful for dynamic control.

### 🔁 Overlapping List Processing

- **`OverlappingImageListSplitter`**  
  Splits a list into up to 6 overlapping image segments.

- **`OverlappingImageListMerger`**  
  Merges overlapping segments back into a continuous list.

### 🎨 Channel Tools

- **`RGBChannelMerger`**  
  Merge three grayscale or channel-separated images into one RGB image.

---

## 🧰 Installation

1. Place the `.py` file in your ComfyUI `custom_nodes` directory.
2. Restart ComfyUI.

