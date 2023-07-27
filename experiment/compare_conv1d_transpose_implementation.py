# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
This script is used to understand Conv1d_transpose operation output during inference by 
comparing keras Conv1DTranspose with the output of the  a diy conv1d_transpose.
"""

def diy_conv1d_transpose_multi(data,kernel,padding="valid",strides=1):
    """
    Immitate basic features of keras conv1d_transpose operation in keras.
    This is for batch>=1 and channels>=1.

    Parameters
    ----------
    data : ndarray
        Data. Shape=(batch,steps,chan)
    kernel : array
        Filter. Shape=(k,) where k is the filter size.
    padding : string, optional
        Padding {valid,same}. The default is "valid".
    strides : int, optional
        Strides. The default is 1.

    Raises
    ------
    ValueError
        If the stride is not allowed.

    Returns
    -------
    ndarray
        The transpose convolution result for each batch and channel.

    """
    
    if strides !=1:
        raise ValueError('Non unitary stride is not supported');
     
    if len(kernel.shape)!=1:
        raise ValueError('Num filters greater than 1 is not supported');
    
    batch_size=data.shape[0];
    chans=data.shape[-1];
    steps=None;
    out_data=None;
    for b in range(batch_size):
        for chan in range(chans):
            # TODO: this will curently not work for num filters greater 
            # than 1 @see diy_conv1d_multi() in compare_conv1d_implementation.py 
            # for implementation that works for multiple channels/features and 
            # multiple filters
            result=diy_conv1d_transpose(data[b,:,chan],kernel,padding=padding);
            if steps is None:
                steps=len(result);
                out_data=np.zeros((batch_size,steps,chans));
            out_data[b,:,chan]=result;
    
    return out_data;

def diy_conv1d_transpose(data,kernel,padding="valid",strides=1):
    """
    Immitate basic features of keras conv1d_transpose operation in keras

    Parameters
    ----------
    data : ndarray
        Data. Shape=(N,)
    kernel : array
        Filter. Shape=(k,) where k is the filter size.
    padding : string, optional
        Padding {valid,same}. The default is "valid".
    strides : int, optional
        Strides. The default is 1.

    Raises
    ------
    ValueError
        If the stride is not allowed.

    Returns
    -------
    ndarray
        The transpose convolution result for each batch and channel.

    """
    
    if strides !=1:
        raise ValueError('Non unitary stride is not supported');
    
    # Compute the input size of Conv1d that would have resulted in data.shape[1]
    s=strides;
    o=len(data);# the output size of the conv1d operation that would lead to our current input size.
    k=kernel.shape[0];#filter_size;
    p=0;# The padding that would have been added to that Conv1d depends on the padding scheme. so lets work it out.
    if padding in ["same"]:
        p=(k-1)//2;
    elif padding in ["valid"]:
        p=0;
    
    # see https://arxiv.org/pdf/1603.07285v1.pdf
    #TODO: check the effect of flooring in the original equation
    i=(o-1)*s - 2*p + k ; #conv1d input size that would have lead to the size of the data we are currently give.
    
    
    # So i must now be the size of our output for our transpose 
    # conv1d. So we must pad our input to produce size i as output size.
    input_size=i;
    output_size=i;
    pad_size= ((output_size-1)*s - input_size + k)/2;
    pad_size=int(np.ceil(pad_size))
    
    
    # Flip the filter
    kernel = np.flipud(kernel);
    data_padded=np.pad(data,pad_size);
    out_data=diy_convolve1d(data_padded,kernel,padding="same");
    
    # When k is even, the output size is suplus by 1.
    if k%2 == 0:
        out_data =out_data[:-1]
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
in_channels = 1
in_length = 4
out_channels = 1
kernel_size = 3
strides = 1

np.random.seed(0);
input_data = np.random.rand(batch_size, in_length,in_channels )
filters = np.random.rand(kernel_size,out_channels, in_channels)

# Step 2: Use your diy conv1d_transpose function
output_diy=diy_conv1d_transpose_multi(input_data,filters[:,0,0])

# Step 3: Create Keras model with Conv1DTranspose
model = tf.keras.Sequential([
    tf.keras.layers.Conv1DTranspose(out_channels, kernel_size, strides=strides, padding='valid', input_shape=(in_length,in_channels),use_bias=False)
])
model.set_weights([filters])
model.summary()


# Step 4: Use Keras model to perform transposed convolution
output_keras = model.predict(input_data)

# Step 5: Compare outputs
print("Output (diy_conv1d_transpose):\n", output_diy)
print("Output (Keras Conv1DTranspose):\n", output_keras)
print("Is it a pass: Is the result approx same")
if np.all(abs(output_diy-output_keras)<0.001):
    print('Passed');
else:
    print('Failed')

# tfdata=tf.constant(input_data,tf.float32);
# tfFilters=tf.constant(filters,tf.float32);
# tf.nn.conv1d_transpose(tfdata,tfFilters,tf.constant([1,in_length,1]),1)
