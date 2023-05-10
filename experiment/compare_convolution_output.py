import keras.backend as K;
import numpy as np;

"""
This script is used to understand Conv1D layer output during inference by 
comparing keras conv1d with the output of the numpy.convolve function.
"""

# Manually implement the convolution operation for padding='same' without using
# the numpy.convolve function.
def convolve(data, kernel):
    # data: 1D array
    # kernel: 1D array
    # output: 1D array
    output = np.zeros(data.shape);
    for i in range(data.shape[0]):
        for j in range(kernel.shape[0]):
            if i-j<0 or i-j>=data.shape[0]:
                continue;
            output[i] += data[i-j]*kernel[j];
    return output;
def convolve1d_same(x, kernel):

    # reverse the kernel
    kernel = np.flipud(kernel);

    # compute padding size
    k_size = kernel.shape[0]
    pad_size = k_size // 2

    # pad size as in tensorflow???
    pad_size=(k_size-1) // 2;
    kernel = np.flipud(kernel);# Re-reverse the kernel to match tensorflow

    # pad the input array
    x_padded = np.pad(x, pad_size, mode='constant')

    

    # initialize output array
    out = np.zeros_like(x)

    # perform convolution
    for i in range(len(out)):
        d=x_padded[i:i+k_size];
        print(f'd={d.shape}');
        if d.shape[0]!=kernel.shape[0]:
            d=np.pad(d, (0, kernel.shape[0]-d.shape[0]), mode='constant');
        out[i] = np.sum(d * kernel)

    return out

# Generate random data and weights
data=np.random.rand(1, 50, 1);
weights = [np.random.rand(12, 1, 1), np.random.rand(1)];

# Perform the convolution operation using numpy
kernels = weights[0];
biases = weights[1];
batch_size = data.shape[0];
data_filtered = np.zeros((batch_size, data.shape[1], kernels.shape[-1]));
for batch_n in range(batch_size):
    for i in range(kernels.shape[-1]):
        data_filtered[batch_n,:,i] = np.convolve(
            data[batch_n,:,0], 
            kernels[:, 0, i], 
            'same') ;
       
np_filtered=data_filtered; # ReLU activation
    
# Perform the same convolution operation using keras.backend
tf_filtered=K.conv1d(
    K.variable(data),
    K.variable(kernels),
    strides=1,
    padding='same',
    data_format="channels_last",
    dilation_rate=1);
#tf_filtered=np.maximum(tf_filtered,0);

# DIY convolution
diy=convolve1d_same(data[0,:,0], kernels[:,0,0]);

# Compare the results
print(np_filtered[0,0:5,0]);
print(tf_filtered[0,0:5,0]);
print(diy[0:5]);
print(np.allclose(np_filtered, tf_filtered));

# Why is the result not the same?