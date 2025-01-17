import tensorflow as tf
import numpy as np

class EagleLoss_tf_3D(tf.keras.losses.Loss):
    def __init__(self, patch_size=3, cutoff=0.5, **kwargs):
        super(EagleLoss_tf_3D, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.cutoff = cutoff

        # Scharr kernels for gradient calculation
        scharr = self.scharr_3d_kernels()
        self.kernel_x = tf.constant(scharr[0], dtype=tf.float32)[...,tf.newaxis,tf.newaxis]
        self.kernel_y = tf.constant(scharr[1], dtype=tf.float32)[...,tf.newaxis,tf.newaxis]
        self.kernel_z = tf.constant(scharr[2], dtype=tf.float32)[...,tf.newaxis,tf.newaxis]
    
    def scharr_3d_kernels(self):
        """ Define the 3D Scharr Kernels to compute the image gradients along x,y,z """
        # TODO: double check the kernels that are used; I'm not 100% certain these are correct for 3D  
        
        # x gradient 
        gx = np.array([[[ 3,  0, -3],
                      [10,  0, -10],
                      [ 3,  0, -3]],
                     [[10,  0, -10],
                      [30,  0, -30],
                      [10,  0, -10]],
                     [[ 3,  0, -3],
                      [10,  0, -10],
                      [ 3,  0, -3]]], dtype=np.float32)
        # y gradient
        gy = np.array([[[ 3, 10,  3],
                      [ 0,  0,  0],
                      [-3, -10, -3]],
                     [[10, 30, 10],
                      [ 0,  0,  0],
                      [-10, -30, -10]],
                     [[ 3, 10,  3],
                      [ 0,  0,  0],
                      [-3, -10, -3]]], dtype=np.float32)
        # z gradient 
        gz = np.array([[[ 3, 10,  3],
                      [10, 30, 10],
                      [ 3, 10,  3]],
                     [[ 0,  0,  0],
                      [ 0,  0,  0],
                      [ 0,  0,  0]],
                     [[-3, -10, -3],
                      [-10, -30, -10],
                      [-3, -10, -3]]], dtype=np.float32)
        
        return [gx, gy, gz]
    
    def rgb_to_grayscale(self, x):
        """Convert RGB image to grayscale using standard weights or average over channels"""
        if x.shape[-1] == 3: # assumes channel last orientation (tf default)
            weights = tf.constant([0.2989, 0.5870, 0.1140], dtype=tf.float32)
            return tf.reduce_sum(x * weights, axis=-1, keepdims=True)
        elif x.shape[-1] > 1: 
            tf.cast(tf.reduce_mean(x, axis=-1, keepdims=True), dtype=tf.float32)
        return x
    
    def call(self, y_true, y_pred):

        y_true = self.rgb_to_grayscale(y_true)
        y_pred = self.rgb_to_grayscale(y_pred)

        # Gradient maps calculation
        gx_pred, gy_pred, gz_pred = self.calculate_gradient(y_pred)
        gx_true, gy_true, gz_true = self.calculate_gradient(y_true)
        
        # Patch-based variance loss
        loss = self.calculate_patch_loss(gx_pred, gx_true) + \
               self.calculate_patch_loss(gy_pred, gy_true) + \
               self.calculate_patch_loss(gz_pred, gz_true) 

        return loss

    def calculate_gradient(self, img):
        """Compute gradients along all 3 dimensions using Scharr filters"""
        gx = tf.nn.conv3d(img, self.kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
        gy = tf.nn.conv3d(img, self.kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
        gz = tf.nn.conv3d(img, self.kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')
        return gx, gy, gz

    def calculate_patch_loss(self, grad_pred, grad_true):
        """Compute the variance-based loss within patches"""
        patch_pred = self.extract_patches(grad_pred)
        patch_true = self.extract_patches(grad_true)
        
        var_pred = tf.math.reduce_variance(patch_pred, axis=-1, keepdims=True)
        var_true = tf.math.reduce_variance(patch_true, axis=-1, keepdims=True)

        return self.fft_loss(var_pred, var_true)

    def extract_patches(self, img):
        """Extract non-overlapping patches from the gradient images"""
        return tf.extract_volume_patches(
            input=img,
            ksizes=[1, self.patch_size, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, self.patch_size, 1],
            padding='VALID'
        )

    def fft_loss(self, pred, target):
        """Calculate the FFT-based high-pass filtered loss"""
        pred_fft = tf.signal.fft3d(tf.squeeze(tf.complex(pred, 0.0), axis=-1))
        target_fft = tf.signal.fft3d(tf.squeeze(tf.complex(target, 0.0), axis=-1))

        pred_mag = tf.abs(pred_fft)
        target_mag = tf.abs(target_fft)
        weights = self.gaussian_highpass_weights3d(pred.shape[1:])
        weighted_pred_mag = weights * pred_mag
        weighted_target_mag = weights * target_mag

        return tf.reduce_mean(tf.abs(weighted_pred_mag - weighted_target_mag))

    def gaussian_highpass_weights3d(self, size):
        """Generate 3D high-pass filter weights in the frequency domain"""
        freq_x = tf.signal.fftshift(tf.range(-size[0] // 2, size[0] // 2, dtype=tf.float32)) / size[0]
        freq_y = tf.signal.fftshift(tf.range(-size[1] // 2, size[1] // 2, dtype=tf.float32)) / size[1]
        freq_z = tf.signal.fftshift(tf.range(-size[2] // 2, size[2] // 2, dtype=tf.float32)) / size[2]

        freq_x = tf.reshape(freq_x, [-1, 1, 1])
        freq_y = tf.reshape(freq_y, [1, -1, 1])
        freq_z = tf.reshape(freq_z, [1, 1, -1])

        freq_mag = tf.sqrt(freq_x ** 2 + freq_y ** 2 + freq_z ** 2)
        weights = tf.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        return 1 - weights  # Inverted for high-pass filtering