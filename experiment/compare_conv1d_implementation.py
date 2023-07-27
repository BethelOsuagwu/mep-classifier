# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
This script is used to understand Conv1d operation output during inference by 
comparing keras Conv1D with the output of the  a diy conv1d operation.
"""

def diy_conv1d_multi(data,kernels,padding="valid",strides=1):
    """
    Immitate basic features of keras conv1d operation in keras.
    This is for batch>=1 and channels>=1.

    Parameters
    ----------
    data : ndarray
        Data. Shape=(batch,steps,chan)
    kernels : array
        Filter. Shape=(k,input_chan,output_chan) where k is the filter size.
    padding : string, optional
        Padding {valid,same}. The default is "valid".
    strides : int, optional
        Strides. The default is 1.

    Returns
    -------
    ndarray
        The convolution result for each batch and channel.

    """
    
    
    batch_size=data.shape[0];
    features=data.shape[-1];#features
    steps=None;
    out_data=None;
    num_filters=kernels.shape[-1];
    for b in range(batch_size):
        for k in range(num_filters):
            for feature in range(features):
                kernel=kernels[:,feature,k]
                result=diy_convolve1d(data[b,:,feature],kernel,padding,strides);
                if steps is None:
                    steps=len(result);
                    out_data=np.zeros((batch_size,steps,num_filters));
                out_data[b,:,k]=out_data[b,:,k]+result;
                
            # If this was layer, we would add bias thus:
            #out_data[b,:,k]=out_data[b,:,k] +bias(k)
    return out_data;



def diy_convolve1d(x, kernel,padding="valid",strides=1):
    """
    Perform elementry convolution as it is defined in deep learning.

    Parameters
    ----------
    x : array
        Data. shape= (N,)
    kernel : array
        1D filter
    padding : int, optional
        The padding scheem {valid,same}. The default is "valid".
    strides : int, optional
        The stride. Onli stride=1 is currently supported. The default is 1.

    Raises
    ------
    ValueError
        If stride is not 1.

    Returns
    -------
    out : array
        1D array. Shape determined by convolution procedure. 

    """
    
    if strides !=1:
        raise ValueError('Non unitary stride is not supported');

    # reverse the kernel
    #kernel = np.flipud(kernel);# This and the re-reverse below should removed as they are reversing each other.

    # compute padding size
    k_size = kernel.shape[0]
   
    #
    s=strides;
    i=len(x);# Length of input
    k=kernel.shape[0];#filter_size;
    p=0;# pading size. This is the size of the padding that is added to each end of the signal. i.e so the total padding=2*p.
    if padding in ["same"]:
       p=(k-1)//2;
       o=i;
    elif padding in ["valid"]:
         p=0;
         #Zero padding, non-unit strides: see https://arxiv.org/pdf/1603.07285v1.pdf
         o=np.floor( (i + 2*p - k)/s) + 1;
         o=int(o);
    
    
    
    #
    #kernel = np.flipud(kernel);# Re-reverse the kernel to match tensorflow

    # pad the input array
    x_padded = np.pad(x, p, mode='constant')

    #For even filter len , we need to add extra 'post' pad when padding is 'same'
    if padding in ["same"]:
        if kernel.shape[0]%2==0:
            x_padded=np.pad(x_padded, (0,1), mode='constant')

    # initialize output array
    out = np.zeros(o);
    
    
            

    # perform convolution
    for i in range(len(out)):
        d=x_padded[i:i+k_size];
        
        #----------------------------
        # This is dirty but it prevents error when the kernel size is 
        # even, and it produces similar results as in Tensorflow.
        # if padding in ["same"]:
        #     if d.shape[0]!=kernel.shape[0]:
        #         d=np.pad(d, (0, kernel.shape[0]-d.shape[0]), mode='constant');
        #----------------------------
        
        out[i] = np.sum(d * kernel)
    
    return out


# ------ MAY THE TEST BEGIN

# Step 1: Create sample input data and filters
batch_size = 1
channels = 2
in_length = 5 #input length
kernel_size = 3
strides = 1
padding='valid';

np.random.seed(0);
input_data = np.random.rand(batch_size, in_length,channels )
filters = np.random.rand(kernel_size,channels, channels)

# Step 2: Use your diy conv1d function
output_diy=diy_conv1d_multi(input_data,filters,strides=strides, padding=padding)

# Step 3: Create Keras model with Conv1D
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(channels, kernel_size, strides=strides, padding=padding, input_shape=(in_length,channels),use_bias=False)
])
model.set_weights([filters])
model.summary()


# Step 4: Use Keras model to perform convolution
output_keras = model.predict(input_data)

# Step 5: Compare outputs
print("Output (diy_conv1d):\n", output_diy)
print("Output (Keras Conv1D):\n", output_keras)
print("Is it a pass: Is the result approx same")
if np.all(abs(output_diy-output_keras)<0.001):
    print('Passed');
else:
    print('Failed')
