import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Set paths to your train and test directories
train_dir = "E:/Desktop/AI/new/dataset/train"
test_dir = "E:/Desktop/AI/new/dataset/test"

# Create ImageDataGenerators for loading and augmenting the images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]  # Adjust brightness
)

test_datagen = ImageDataGenerator(rescale=1./255)

# # Flow images from the directory into the model
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

test_generator = test_datagen.flow_from_directory(
   test_dir,
   target_size=(150, 150),  # Resize images to 150x150
   batch_size=32,
    class_mode='categorical'  # For multi-class classification
)


# Load the trained model
model = tf.keras.models.load_model("animal_classifier_model.h5")

# Load class labels
class_names = open("./labels.txt", "r").readlines()  # Ensure labels.txt is in the correct path

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))  # Ensure the image size matches the training size
    img_array = image.img_to_array(img) #img_array has the shape (height, width, channels), representing pixel values.
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension #the model expects a 4D input (batch size, height, width, channels).
    img_array = img_array / 255.0  # Normalize the image

    # Predict the class
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)  # Get the index of the highest probability
    predicted_class = class_names[class_index[0]].strip()  #The .strip() function is used to remove any leading or trailing whitespace from the class name.
 
    print(f"Predicted Class: {predicted_class}")
    return predicted_class

# Example usage

img_path = "pexels-photo-1108099.jpeg" 
predicted_class = predict_image(img_path)
print(f"Result: {predicted_class}")

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# # Visualization of Training History (Loss and Accuracy)
# import matplotlib.pyplot as plt

# Plot training & validation accuracy values
# Visualization of Training History (Loss and Accuracy)
# import matplotlib.pyplot as plt

# # Plot training & validation accuracy values
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Test'], loc='upper left')


import matplotlib.pyplot as plt
import numpy as np

# Evaluate on training and test data
train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

# Plot training and validation accuracy values
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.bar(["Train", "Test"], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Loss plot
plt.subplot(1, 2, 2)
plt.bar(["Train", "Test"], [train_loss, test_loss], color=['blue', 'orange'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.ylim(0, max(train_loss, test_loss) + 0.1)

plt.show()
