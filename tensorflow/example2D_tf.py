import tensorflow as tf
from tensorflow.keras import layers, models
from eagle_loss_tf_2D import EagleLoss_tf_2D
disp = False # Display flag 

# Simple AE model
def create_denoising_autoencoder():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),    # Input shape for MNIST images
        layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

# Load dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]

# Add noise to the images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# Clip the values to stay within [0, 1]
x_train_noisy = tf.clip_by_value(x_train_noisy, 0.0, 1.0)
x_test_noisy = tf.clip_by_value(x_test_noisy, 0.0, 1.0)

# Reshape data for the model
x_train = x_train[..., tf.newaxis]  # Add channel dimension
x_test = x_test[..., tf.newaxis]
x_train_noisy = x_train_noisy[..., tf.newaxis]
x_test_noisy = x_test_noisy[..., tf.newaxis]

# Create the model
model = create_denoising_autoencoder()

# ------------------------------
# define the eagle loss 
custom_loss = EagleLoss_tf_2D(patch_size=3, cutoff=0.5)
# ------------------------------

# Compile model
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
history = model.fit(x_train_noisy, x_train, 
                    validation_data=(x_test_noisy, x_test),
                    epochs=5, batch_size=32)

# Evaluate the model
loss = model.evaluate(x_test_noisy, x_test)
print(f"Test Loss: {loss}")

if disp: 
  # Display some results
  import matplotlib.pyplot as plt
  
  # Visualize the noisy vs. denoised images
  n = 10  # Number of images to display
  denoised_images = model.predict(x_test_noisy[:n])
  
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # Display noisy image
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(x_test_noisy[i].numpy().squeeze(), cmap='gray')
      plt.title("Noisy")
      plt.axis("off")
      
      # Display denoised image
      ax = plt.subplot(2, n, i + n + 1)
      plt.imshow(denoised_images[i].squeeze(), cmap='gray')
      plt.title("Denoised")
      plt.axis("off")
  plt.show()
