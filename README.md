<!--
 * @Author: your name
 * @Date: 2021-11-02 15:55:09
 * @LastEditTime: 2021-11-02 16:18:30
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Fish-Brain-Behavior-Analysis/code/dual_Color_pipeline/README.md
-->
# Dual-Color Image Processing Pipeline

This repository contains a dual-color image processing pipeline implemented in Matlab. The pipeline accomplishes the extraction and correction of whole-brain calcium activity in freely swimming zebrafish by processing data acquired by a two-color light field imaging system. Please check our paper "[All-optical interrogation of brain-wide activity in freely swimming larval zebrafish](https://www.biorxiv.org/content/10.1101/2023.05.24.542114v1)" for more details.

## Pipeline
![pipeline](https://github.com/Wenlab/Fish-Brain-Behavior-Analysis/assets/19462042/b09a17cd-308f-4c45-9c39-a437efea1072)

The pipeline consists of the following main steps:
- [Dual-color 3D Image Reconstruction](https://github.com/Wenlab/Dual-Color-Image-Processing/tree/main/Reconstruction)
- [Image Registration](https://github.com/Wenlab/Dual-Color-Image-Processing/tree/main/Registration)
- [Region Segmentation and Signal Extraction](https://github.com/Wenlab/Dual-Color-Image-Processing/tree/main/Segmentation-Extraction)
- [Adaptive Filter Calcium Signal Correction](https://github.com/Wenlab/Dual-Color-Image-Processing/tree/main/AdaptiveFilter)



