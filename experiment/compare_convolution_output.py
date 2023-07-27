import keras.backend as K;
import numpy as np;

"""
This script is used to understand Conv1D layer output during inference by 
comparing keras conv1d with the output of the numpy.convolve function.
"""

def conv1d_transpose(data,kernel,padding="valid",strides=1):
    
    if strides !=1:
        raise ValueError('Non unitary stride is not supported');
    
    # if padding != "valid":
    #     raise ValueError('Only "valid" is supported for padding');
           
    
    # compute the input size of Conv1d that will result in data.shape[1]
    s=strides;
    o=data.shape[1];# the output size of the conv1d operation that would lead to our current input size.
    k=kernel.shape[0];#filter_size;
    p=0;# pading size
    if padding in ["same"]:
       p=(k-1)//2;
    elif padding in ["valid"]:
         p=0;
    
    # see https://arxiv.org/pdf/1603.07285v1.pdf
    #TODO: check the effect of flooring in the original equation
    i=(o-1)*s - 2*p + k ; #conv1d input size that would have lead to the size of the data we are currently give.
    
    #print(f"dhdh:{i}\n p={p}")
    #return
    
    # So i must now be the size of our output for our transpose 
    # conv1d. So we must pad our input to produce size i as output size.
    input_size=i;
    output_size=i;
    pad_size= ((output_size-1)*s - input_size + k)/2;
    pad_size=int(np.ceil(pad_size))
    
    #Do the padding
    #data_padded = np.pad(data, pad_size, mode='constant');
    
    # Flit the filter
    kernel = np.flipud(kernel);
    
    # Call conv 1d
    batch_size=data.shape[0];
    chans=data.shape[-1];
    steps=data.shape[1];
    out_data=np.zeros((batch_size,steps+pad_size*2,chans));
    for b in range(batch_size):
        for chan in range(chans):
            d=np.pad(data[b,:,chan],pad_size);
            #print(f"lioik:{pad_size}")
            out_data[b,:,chan]=convolve1d(d,kernel,padding="same");
    return out_data;

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
    #pad_size = k_size // 2

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
        #print(f'd={d.shape}');
        if d.shape[0]!=kernel.shape[0]:
            d=np.pad(d, (0, kernel.shape[0]-d.shape[0]), mode='constant');
        out[i] = np.sum(d * kernel)

    return out

def convolve1d(x, kernel,padding="valid",strides=1):
    
    if strides !=1:
        raise ValueError('Non unitary stride is not supported');
        
    # reverse the kernel
    kernel = np.flipud(kernel);

    # compute padding size
    k_size = kernel.shape[0]
   
    #
    s=strides;
    i=len(x);# Length of input
    k=kernel.shape[0];#filter_size;
    p=0;# pading size
    if padding in ["same"]:
       p=(k-1)//2;
    elif padding in ["valid"]:
         p=0;
    
    #Zero padding, non-unit strides=> see https://arxiv.org/pdf/1603.07285v1.pdf
    o=np.floor( (i + 2*p - k)/s) + 1;
    o=int(o);
   
    
    #
    kernel = np.flipud(kernel);# Re-reverse the kernel to match tensorflow

    # pad the input array
    x_padded = np.pad(x, p, mode='constant')

    

    # initialize output array
    out = np.zeros(o);

    # perform convolution
    for i in range(len(out)):
        d=x_padded[i:i+k_size];
        #print(f'd={d.shape}');
        if padding in ["same"]:
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