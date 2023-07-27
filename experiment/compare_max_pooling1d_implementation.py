# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
This script is used to understand max pooling operation output during inference by 
comparing keras max pooling 1d with the output of the  a diy max pooling.
"""

def diy_MaxPooling1D(data,pool_size,padding="valid",strides=1):
    """
    Immitate basic features keras max pooling 1d. But ofcourse this is not layer.

    Parameters
    ----------
    data : ndarray
        Data. Shape=(batch,steps,chan)
    pool_size : int
        Pool size.
    padding : string, optional
        Padding {valid,same}. The default is "valid".
    strides : int, optional
        Strides. The default is 1.

    Raises
    ------
    ValueError
        If the given strides,/padding option is not allowed.

    Returns
    -------
    ndarray
        The max pooling result along the steps(time) axis.

    """
    
    if strides <1:
        raise ValueError('Strides cannot be less than 1');
    if padding !='valid':
        raise ValueError('The given padding option is implemented')
    
    input_steps=data.shape[1];
    output_steps = ((input_steps - pool_size ) / strides)+1; #https://arxiv.org/pdf/1603.07285v1.pdf # But a wrong foumula @see keras.layers.MaxPooling1D
    output_steps=int(output_steps);
    
    batch_size=data.shape[0];
    chans=data.shape[-1];
    out_data=np.zeros((batch_size,output_steps,chans));
    for b in range(batch_size):
        for chan in range(chans):
            outstep=-1;
            for instep in range(0,(input_steps-pool_size+1),strides):

                result=data[b,instep:instep+pool_size,chan];
                outstep=outstep+1;# OR: outstep=int(instep/strides);
                
                result=np.max(result);
                out_data[b,outstep,chan]=result;
            
    
    return out_data;


# ------ MAY THE TEST BEGIN

# Step 1: Create sample input data and filters
batch_size = 1
chans = 1
steps = 38
pool_size = 2
strides = 2

np.random.seed(0);
input_data = np.random.rand(batch_size, steps,chans )

# Step 2: Use your diy function
output_diy=diy_MaxPooling1D(input_data,pool_size,'valid',strides)

# Step 3: Create Keras model 
model = tf.keras.Sequential([
    tf.keras.layers.MaxPooling1D(pool_size, strides=strides, padding='valid')
]);
model.build(input_shape=input_data.shape);
model.summary()


# Step 4: Use Keras model to perform task
output_keras = model.predict(input_data)

# Step 5: Compare outputs
print("Output (diy):\n", output_diy)
print("Output (Keras ):\n", output_keras)
print("Is it a pass?: Is the result approx same?")
if np.all(output_diy.shape==output_keras.shape) and np.all(abs(output_diy-output_keras)<0.001):
    print('Passed');
else:
    print('Failed')
