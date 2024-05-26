# Plot the validation and training curves separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics
  """
  # Extract the training loss values from the training history
  loss = history.history['loss']

  # Extract the validation loss values from the training history
  val_loss = history.history['val_loss']

  # Extract the training accuracy values from the training history
  accuracy = history.history['accuracy']

  # Extract the validation accuracy values from the training history
  val_accuracy = history.history['val_accuracy']

  # The total of epochs
  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()
  plt.show()

  # Plot accuracy
  plt.figure() # to separate two plots
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('accuracy')
  plt.xlabel('epochs')
  plt.legend()
  plt.show()
