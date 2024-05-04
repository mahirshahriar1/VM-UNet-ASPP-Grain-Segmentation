# Project Demo 2

## Papers We Worked With
- [Grain and Grain Boundary Segmentation using Machine Learning with Real and Generated Datasets](https://arxiv.org/pdf/1906.11031)
- [Continuous Dice Coefficient: a Method for Evaluating Probabilistic Segmentations](https://arxiv.org/pdf/1906.11031)
- [Segment Anything](https://arxiv.org/pdf/2304.02643)
- [CONVOLUTION MEETS LORA: PARAMETER EFFICIENT FINETUNING FOR SEGMENT ANYTHING MODEL](https://arxiv.org/pdf/2401.17868)
- [Customized Segment Anything Model for Medical Image Segmentation](https://arxiv.org/pdf/2304.13785)
- [UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer](https://arxiv.org/pdf/2109.04335)
- [VM-UNET-V2 Rethinking Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/pdf/2403.09157)
- [VM-UNet: Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/pdf/2402.02491)
- [UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer](https://arxiv.org/pdf/2109.04335)
- [Medical Image Segmentation Review: The success of U-Net](https://arxiv.org/pdf/2211.14830)


## Folder Structure

### SAM
- `fine_tune_SAM_grains`: Contains the fine-tuned SAM model for grains. **Dice Score - (88.4%)**

### SAM LoRA
- `SAM_LORA (15 epoch 20 image)`: SAM LoRA model trained for 15 epochs on 20 images. **Dice Score - (90.1%)**
- `SAM_LORA (15 epoch 408 image)`: SAM LoRA model trained for 15 epochs on 408 images. **Dice Score - (90.9%)**

### Vision Mamba UNet
- `VM_UNet_Grain_Segmentation`: Vision Mamba UNet
- `VM-UNetV2 Grain Segmentation`: Vision Mamba UNet V2

### U-Net Variants UCTransNet DeepLabv3+
- `Ensemble_(spittted)`: Containing SMP models combinations. **Dice Score - (90%)**
- `Unet_att`: U-Net model with attention mechanism. **Dice Score - (89.5%)**
- `UCTransnet`: Contains the UCTransNet model with combined_dice_ce_loss. **Dice Score - (90.1%)**
- `UCTransnet_1_`: Contains the UCTransNet model with custom DiceLoss. **Dice Score - (91.05%)**
- `Unet++_resnet152_cDCloss`: U-Net++ model with ResNet-152 backbone and with combined_dice_ce_loss. **Dice Score - (90.6%)**
- `DeeplabV3_cDC`: DeeplabV3+ model with different encoders **Dice Score - (89.66%)**

### Unet_Base Model
- `Unet_Base_Model`: Base U-Net model. **Dice Score - (90.66%)**
- `Unet_Base_Model (Trained on 20 image)`: Base U-Net model trained on 20 images. **Dice Score - (89.04%)**
- `R2AttU`: R2AttU_Net_with_Sigmoid

![Dice Scores for Various Models](/Images/Demo2.jpg)
