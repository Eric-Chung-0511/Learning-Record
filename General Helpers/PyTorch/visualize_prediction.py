import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def visualize_predictions(model, test_data, class_names, device, num_samples=10):
    """
    Visualize random predictions made by the model on test data.
    
    Parameters:
    - model: The trained model for inference
    - test_data: The test dataset
    - class_names: List of class names for predictions
    - device: Device to perform inference ('cpu' or 'cuda')
    - num_samples: Number of samples to visualize (default is 10)
    """
    plt.figure(figsize=(25, 10))  

    for i in range(num_samples):
        # random pick samples
        idx = random.randint(0, len(test_data)-1)
        img, label = test_data[idx]
        
        # model inference
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred_label = output.argmax(dim=1).item()
        
        img_np = img.permute(1, 2, 0).cpu().numpy()  # turn into matplotlib format
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean  # reverse the standardization
        img_np = np.clip(img_np, 0, 1)  # limit to the range(0, 1)

        # show image
        ax = plt.subplot(2, 5, i+1)  
        plt.imshow(img_np)
        plt.axis('off')
        
        # if the prediction is wrong use red word, otherwise use green word
        pred_color = 'red' if pred_label != label else 'green'
        
        # adjust the text above the image
        ax.text(0.5, 1.10, f"True: {class_names[label]}", 
                color='green', ha='center', va='bottom', transform=ax.transAxes, fontsize=18)
        ax.text(0.5, 1.02, f"Pred: {class_names[pred_label]}", 
                color=pred_color, ha='center', va='bottom', transform=ax.transAxes, fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.6) 
    plt.show()
