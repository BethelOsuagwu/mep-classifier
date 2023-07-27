# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
This script is used to understand upsampling 1d operation output during inference by 
comparing keras upsampling with the output of the  a diy upsampling 1d.
"""

def diy_Upsampling1D(data,size):
    """
    Immitate basic operation of UpSampling1D layer in keras. But ofcourse this 
    is not a layer. The upsampling is along the timesteps axis.

    Parameters
    ----------
    data : ndarray
        Data. 2D array of Shape=(batch,timesteps,chans).
    size : int
        Upsampling rate/scale. The default is 2.

    Returns
    -------
    ndarray
        The channelwise upsampled version of the data.

    """
    d=None;
    for n in range(len(data)):
        row=data[n,:];
        rows=np.repeat(row,size,axis=0);
        rows=np.expand_dims(rows,axis=0);
        
        
        if d is None:
            d=rows;
        else:
            d=np.concatenate((d,rows),axis=0);
        
    return d;


# ------ MAY THE TEST BEGIN

# Step 1: Create sample input data and filters
batch_size = 2
chans = 4
timesteps=3;
upsample_size=2;

np.random.seed(0);
input_data = np.random.rand(batch_size, timesteps,chans )

# Step 2: Use your diy function
output_diy=diy_Upsampling1D(input_data, upsample_size);

# Step 3: Create Keras model 
model = tf.keras.Sequential([
    tf.keras.layers.UpSampling1D(upsample_size)
])
model.build(input_shape=input_data.shape)
model.summary()


# Step 4: Use Keras model to perform task
output_keras = model.predict(input_data)

# Step 5: Compare outputs
print("Output (diy):\n", output_diy)
print("Output (Keras ):\n", output_keras)
print("Is it a pass: Is the result approx same")
if np.all((output_diy-output_keras)<0.001):
    print('Passed');
else:
    print('Failed')

