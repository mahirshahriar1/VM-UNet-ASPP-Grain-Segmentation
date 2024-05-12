# https://www.tutorialspoint.com/execute_matplotlib_online.php
import matplotlib.pyplot as plt

# Reversed data for plotting
model_names = [
    "R2AttU-Net",
    "VM-UNET",
    "VM-UNET-V2",
    "DeepLabV3Plus (Encoder - resnet152)",
    "Ensemble Segmentation Models (Unet++ with pretrained encoders)",
    "Unet++ resnet152 cDCloss", 
    "UCTransnet (DiceLoss)", 
    "UCTransnet (combined_dice_ce_loss)", 
    "Attention U-Net",
    "SAM LoRA (15 epoch 408 image)",
    "SAM LoRA (15 epoch 20 image)",
    "Fine Tune SAM",
    "Unet Base Model (Trained on 20 image)", 
    "Unet Base Model"
]

dice_scores = [
    90.2,
    91.82,
    90.5,
    89.66,
    90, 
    90.6, 
    91.05, 
    90.1, 
    89.5, 
    90.9, 
    90.1, 
    88.4, 
    89.04, 
    90.66
]

# Plotting
plt.figure(figsize=(14, 8))  # Set the figure size
bars = plt.barh(model_names, dice_scores, color='skyblue')  # Horizontal bar plot
plt.xlabel('Dice Score (%)', fontsize=14)
plt.title('Dice Scores for Various Models', fontsize=16)
plt.xlim(85, 92)  # Set the x-axis limit to match the user's graph style

# Add grid with decimal values on the x-axis
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}%'.format(x)))
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid to the x-axis
plt.tight_layout()  # Adjust layout

# Annotate each bar with the respective dice score
for bar, score in zip(bars, dice_scores):
    plt.text(
        bar.get_width(),  # X position for text
        bar.get_y() + bar.get_height() / 2,  # Y position for text
        f' {score}%',  # Text to display
        va='center',  # Vertical alignment
        ha='left',  # Horizontal alignment
        fontsize=10
    )

# Save the figure
plt.savefig('reversed_model_dice_scores_with_grid_and_labels.png')  # Save the plot as a PNG file
plt.show()  # Display the plot
