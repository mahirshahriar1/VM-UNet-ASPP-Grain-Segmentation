# Repository Title

## Folder Structure

### SAM
- `fine_tune_SAM_grains`: Contains the fine-tuned SAM model for grains. **Dice Score - (87.9%)**

### SAM LoRA
- `SAM_LORA (15 epoch 20 image)`: SAM LoRA model trained for 15 epochs on 20 images. **Dice Score - (90.1%)**
- `SAM_LORA (15 epoch 408 image)`: SAM LoRA model trained for 15 epochs on 408 images. **Dice Score - (90.9%)**

### U-Net Variants and UCTransNet
- `Ensemble_(spittted)`: Containing SMP models combinations. **Dice Score - (90%)**
- `Unet_att`: U-Net model with attention mechanism. **Dice Score - (89.5%)**
- `UCTransnet`: Contains the UCTransNet model with combined_dice_ce_loss. **Dice Score - (90.1%)**
- `UCTransnet_1_`: Contains the UCTransNet model with custom DiceLoss. **Dice Score - (91.05%)**
- `Unet++_resnet152_cDCloss`: U-Net++ model with ResNet-152 backbone and with combined_dice_ce_loss. **Dice Score - (90.6%)**

### Unet_Base Model
- `Unet_Base_Model`: Base U-Net model. **Dice Score - (90.66%)**
- `Unet_Base_Model (Trained on 20 image)`: Base U-Net model trained on 20 images. **Dice Score - (89.04%)**

![Dice Scores for Various Models](/Graph.png)
