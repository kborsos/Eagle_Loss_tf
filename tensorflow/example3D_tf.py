import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from eagle_loss_tf_3D import EagleLoss_tf_3D

# Simplest trainable 3D example (toy problem) 

def simple_3d_image_to_image_cnn(input_shape=(32, 32, 32, 1)):
    """ Creates a super-simple 3D CNN for image-to-image tasks. """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    outputs = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
    # Create the model
    model = tf.keras.Model(inputs, outputs)
    return model

# Example usage
input_shape = (32, 32, 32, 1)  # Shape of input data
model = simple_3d_image_to_image_cnn(input_shape=input_shape)

# Define the Eagle Loss 
custom_loss = EagleLoss_tf_3D(patch_size=3,cutoff=0.3)

# Display the model summary
model.summary()
model.compile(optimizer='adam', loss=custom_loss)

# Example training data
x_train = tf.cast(np.random.rand(10, 32, 32, 32, 1), tf.float32)  # 10 random noisy 3D samples
y_train = tf.cast(np.random.rand(10, 32, 32, 32, 1), tf.float32)  # 10 random clean 3D samples (e.g., ground truth)

# Train the model
model.fit(x_train, y_train, batch_size=2, epochs=5)