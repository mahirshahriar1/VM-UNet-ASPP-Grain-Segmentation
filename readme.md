# VM-UNet-ASPP: Vision Mamba UNet with Atrous Spatial Pyramid Pooling for Multi-Feature Extraction in Grain Segmentation

## Introduction

Grain segmentation in materials such as stainless steel is crucial for industrial applications like quality control and material characterization. Accurate delineation of grain structures is essential for understanding material properties. This project presents an enhanced VM-UNet architecture integrated with Atrous Spatial Pyramid Pooling (ASPP) to significantly improve segmentation accuracy of grain structures in stainless steel.


## Dataset used from this paper
- **Title**: Grain and Grain Boundary Segmentation using Machine Learning with Real and Generated Datasets
- **Authors**: Peter Warren, Nandhini Raju, Abhilash Prasad, Shajahan Hossain, Ramesh Subramanian, Jayanta Kapat, Navin Manjooran, Ranajay Ghosh
- **Paper**: [arXiv:2307.05911](https://arxiv.org/abs/2307.05911)

## Overview

### Key Features

- **ASPP Integration**: Enhances the VM-UNet model by capturing multi-scale features without losing resolution, crucial for segmenting grains of varying scales and shapes.
- **High Precision**: Achieves a Dice Score of 91.95%, outperforming traditional models like U-Net, SegNet, and state-of-the-art architectures such as SAM.
- **Robust Performance**: Maintains high performance under challenging conditions, demonstrating reliability in diverse segmentation scenarios.

## Methodology

### Model Architecture

The core of our approach is the integration of the Atrous Spatial Pyramid Pooling (ASPP) block into the VM-UNet architecture. This enhancement allows the model to capture intricate grain details across multiple scales, improving segmentation accuracy.


## Results

The enhanced VM-UNet with ASPP achieved the highest Dice Score of 91.95%, surpassing other models such as U-Net, UCTransNet, and SAM. Key observations include:

- **VM-UNet ASPP vs. VM-UNet V2 with Edge Aware Attention**: The ASPP-enhanced model outperforms VM-UNet V2 by 0.22%.
- **Performance Comparison**: Traditional models like U-Net and UCTransNet achieve Dice Scores between 90.0% and 91.05%, while VM-UNet ASPP excels with 91.95%.
- **Generalist Models**: SAM and SAM LoRA show respectable performance but fall short compared to the specialized VM-UNet ASPP.

## Conclusion

The integration of ASPP into the VM-UNet architecture significantly enhances grain segmentation accuracy in stainless steel microstructures. This advancement paves the way for more reliable and versatile segmentation solutions in material science applications.

### Future Work

1. **Dataset Diversity**: Evaluate model performance on more diverse datasets to ensure robustness and generalizability.
2. **Model Refinement**: Further refinement through advanced techniques like transfer learning and data augmentation.

## Contact

For any questions or feedback, please reach out to:

- [Mahir Shahriar Tamim](mailto:mahir.tamim@northsouth.edu)
- [Fuwad Hasan](mailto:fuwad.hasan@northsouth.edu)
- [Meharun Nesa](mailto:meharun.nesa@northsouth.edu)
