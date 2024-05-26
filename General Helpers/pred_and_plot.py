def pred_and_plot(model, filename, class_name=class_name):
  """
 Imports an image located at the filename, makes a prediction with the model, and plots the image with the predicted class as the title.

`pred` is the prediction result from the model. Assuming your model is a binary classification model, `pred` will be a tensor of shape (1, 1) 
 containing a single prediction value. For example, `pred` might be [[0.8]].

`tf.round(pred)` rounds the prediction value. `tf.round([[0.8]])` will result in [[1.0]].

`tf.round(pred)[0][0]` extracts the numerical value from the prediction result. `tf.round(pred)` is [[1.0]], so `tf.round(pred)[0][0]` is 1.0.

`int(tf.round(pred)[0][0])` converts the float 1.0 to the integer 1.

`class_names[int(tf.round(pred)[0][0])]` uses the integer 1 as an index to find the corresponding class name in the `class_names` list.

  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make prediction
  pred = model.predict(img)

  # Get the predicted class
  pred_class = class_name[int(tf.round(pred)[0][0])]

  # Remove batch dimension for plotting
  img = tf.squeeze(img, axis=0)

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f'Prediction: {pred_class}')
  plt.axis(False);
