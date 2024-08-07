import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *
from keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

# Load the data using the custom load_data function from utils.py
# The size parameter is set to '28', which means the images will be 28x28 pixels
train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(size='28')

# Print the shapes of the loaded data
print(f'The shape of Training images: {train_images.shape}')
print(f'The shape of Training labels: {train_labels.shape}\n')
print(f'The shape of Validation images: {val_images.shape}')
print(f'The shape of Validation labels: {val_labels.shape}\n')
print(f'The shape of Test images: {test_images.shape}')
print(f'The shape of Test labels: {test_labels.shape}')

# Print the number of instances for each class in the training set
print(f'The number of class 0 (akiec): {sum(train_labels == 0).item():3}')
print(f'The number of class 1 (bcc): {sum(train_labels == 1).item():5}')
print(f'The number of class 2 (bkl): {sum(train_labels == 2).item():5}')
print(f'The number of class 3 (df): {sum(train_labels == 3).item():5}')
print(f'The number of class 4 (nv): {sum(train_labels == 4).item():6}')
print(f'The number of class 5 (mel): {sum(train_labels == 5).item():6}')
print(f'The number of class 6 (vasc): {sum(train_labels == 6).item():3}')

# Normalize the images by dividing by 255.0
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),  # Adjusted input_shape to match the new size
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)  # Assuming you have 10 classes
])

# Print the summary of the model
model.summary()

# Set the learning rate
learning_rate = 0.001

# Compile the model with Adam optimizer, SparseCategoricalCrossentropy loss and accuracy as the metric
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    batch_size=2048,
    epochs=22,
    verbose=2
)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the figure
plt.savefig('training_metrics.png')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(
    x=test_images,
    y=test_labels,
    batch_size=16,
    verbose=1,
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=False,
)

# Print the test loss and accuracy
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)