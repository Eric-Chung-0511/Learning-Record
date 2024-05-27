# pred_probs.argmax(): This method returns the index of the maximum value in the pred_probs array. 
# Suppose pred_probs is the array of predicted probabilities from the model, such as [0.1, 0.2, 0.05, 0.15, 0.5]. 
# The argmax() method will return the index of the highest probability, which is index 4.

# classes[pred_probs.argmax()]: Use the above index to get the corresponding class name from the classes list. 
# Suppose classes is ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], 
# then classes[4] will return 'Coat'.
# Use this index to get the corresponding class name from the classes list. 
# For example, if the true label index is 1, it will return "Trouser".
import matplotlib.pyplot as plt
import random

def plot_random_image(model, images, true_labels, classes):

  # Set up random integer
  i = random.randint(0, len(images))

  # Create predictions and targets
  target_image = images[i]

  # have to reshape to get into right size for model,because the input_shape is (28,28)
  # When input_size(28,28), it actually means (1,28,28), (batch_size, weighth, hieghth), 1 image at time
  pred_probs = model.predict(target_image.reshape(1, 28, 28))
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  # Plot image
  plt.imshow(target_image, cmap=plt.cm.binary)

  # Change the color of the titles depending on if the prediction is right or wrong
  if pred_label == true_label:
    color = 'green'
  else:
    color = 'red'

  # Add xlabel information (prediction/ true label)
  plt.xlabel('Pred: {} {:2.0f}% (True: {})'.format(pred_label, 100*tf.reduce_max(pred_probs), true_label), color=color)
