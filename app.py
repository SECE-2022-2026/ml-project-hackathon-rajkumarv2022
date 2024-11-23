import tensorflow as tf
from tensorflow.keras import layers, models
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#sequentioa - 
#Dense-create nuron
#dropout-avoid model overfit
#rmsprop-error preventing
#keranal will be choosed based on image pixels
#kernal small matrix first layer of cnn
# filter collection of kernals
#padding-at the edge add zeros for focsuing on central pixel
#stride - leving shells col and row vise
#weight (28X28X3)^2
#pooling extracting values by max pooling avg pooling transfer poolim

# # Set paths to your train and test directories
train_dir = "E:/Desktop/AI/new/dataset/train"
test_dir = "E:/Desktop/AI/new/dataset/test"

# # Create ImageDataGenerators for loading and augmenting the images
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

test_datagen = ImageDataGenerator(rescale=1./255) #  0,1

# # # Flow images from the directory into the model
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

# # # Load the VGG16 model pre-trained on ImageNet and exclude the top layer
#kernal 3X3
# vgg stride=1
#padding same
#max pooling 2x2

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# # # Freeze the base model to prevent training of pre-trained weights
base_model.trainable = False

# # # Build the model using VGG16 as the base and adding custom layers
#Flatten is a layer used to reshape multi-dimensional inputs into a 1D vector (a flat array).
# Rectified Linear Unit 
#biass actviation fun hiddenlayer

#sigmoid tanh relu leaky rely

#muyltiply input and kernal and store it in a single shell kernal can be 2X2 3X3 padding-adding leading zero stride-leving two row and two col 
#pooling for etracting main features maxpoll,avg pool, transfer pool

model = models.Sequential([
    base_model,
    layers.BatchNormalization(),  # Add Batch Normalization
    layers.Flatten(), #Flattening (multi dim to 1 d vector)  an image refers to the process of converting a multi-dimensional image (such as a 2D image or 3D array) into a one-dimensional array (vector). 
    layers.Dense(512, activation='relu'), # creating nuron
    layers.Dropout(0.5),  # Add Dropout to avoid overfitting 
    layers.Dense(256, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# # # Compile the model
# Adaptive Moment Estimation
# Momentum
# Adaptive Learning Rate: A  # sparse gradients # excluding zeroes
#Measures the difference between the predicted and actual outputs.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
#The loss function tries to minimize the difference between the predicted probability distribution and the actual distribution
#RMSProp.
#It adapts the learning rate based on the first and second moments
#it determines the size of the steps the model takes when updating its parameters during the optimization process.

# # # Callbacks for early stopping and learning rate reduction
#This callback stops the training early if the model's performance on the validation set stops
#Waits for 5 epochs 
#plateaus refer to a point during training when the performance metric being monitored 

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# # # Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping, lr_reducer]
)

# # Save the model
model.save("animal_classifier_model.h5")

#The VGG16 model is pre-trained and used as a feature extractor.
# own dataset to fine-tune the model and perform classification for specific task.

## edges, textures, shapes

# # # Evaluate the model
# score = model.evaluate(test_generator)
# print(f"Test Loss: {score[0]}, Test Accuracy: {score[1]}")

# # # Visualization of Training History (Loss and Accuracy)
# # import matplotlib.pyplot as plt

# # # Plot training & validation accuracy values
# # plt.figure(figsize=(12, 6))
# # plt.subplot(1, 2, 1)
# # plt.plot(history.history['accuracy'])
# # plt.plot(history.history['val_accuracy'])
# # plt.title('Model accuracy')
# # plt.xlabel('Epoch')
# # plt.ylabel('Accuracy')
# # plt.legend(['Train', 'Test'], loc='upper left')

# # # Plot training & validation loss values
# # plt.subplot(1, 2, 2)
# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.title('Model loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend(['Train', 'Test'], loc='upper left')

# # plt.show()

