import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  # For reading images
import random  # For generating random numbers
import os  # For file and directory operations

def view_random_image(target_dir, target_class):
    """
    Visualize a random image from a specified directory and class.

    Parameters:
    - target_dir (str): The directory where image classes are stored.
    - target_class (str): The class folder to select a random image from.

    Returns:
    - img (ndarray): The randomly selected image array.
    """
    # Setup the target directory
    target_folder = target_dir + target_class
    # Get a random image path, randomly sample one of the items in target_folder
    random_image = random.sample(os.listdir(target_folder), 1)
    print(random_image)
    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)  # Display the image
    plt.title(target_class)  # Set the title of the plot to the target class
    plt.axis('off')  # Turn off the axis
    # Show the shape of the image
    print(f'Image shape: {img.shape}')
    return img  # Return the image

# Example usage:
# img = view_random_image('path/to/dataset/', 'class_name')
