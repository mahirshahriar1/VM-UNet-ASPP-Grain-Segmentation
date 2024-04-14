# Improved Grain Segmentation

## Project Overview
This project aims to enhance the accuracy of grain segmentation algorithms by refining an existing model described in the referenced paper. The goal is to develop a model that generalizes well across various types of grain image datasets, improving robustness and applicability in real-world scenarios.

## Reference
- **Title**: Grain and Grain Boundary Segmentation using Machine Learning with Real and Generated Datasets
- **Authors**: Peter Warren, Nandhini Raju, Abhilash Prasad, Shajahan Hossain, Ramesh Subramanian, Jayanta Kapat, Navin Manjooran, Ranajay Ghosh
- **Paper**: [arXiv:2307.05911](https://arxiv.org/abs/2307.05911)

## Datasets
The project utilizes several datasets comprising different grain image types and segmentation masks. These datasets include manually segmented images, artificially generated images through methods like Voronoi Tessellation, and images processed with various preprocessing techniques like HED (Holistically-Nested Edge Detection), GRAD (Gradient-based preprocessing), and THRESHOLD preprocessing.

### Dataset Configuration
- **Training Set 1**: 100% Manually Segmented
- **Training Set 2**: 25% Artificial (Voronoi Tessellation), 75% Manually Segmented
- More detailed configurations are described in the [New Dataset Works](readme.md).

## Methodology
The methodology involves:
- Evaluating the base model's performance on each type of dataset.
- Implementing enhancements such as advanced preprocessing techniques and neural network adjustments.
- Cross-validating the results to ensure consistency and reliability of the segmentation improvements.

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repository-url
   cd your-repository-directory