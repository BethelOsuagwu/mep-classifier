from tensorflow import keras;
import numpy as np;

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
recurrent_activation=sigmoid;
activation=np.tanh;

def rnn_cell(inputs, prev_state, prev_carry, kernel, recurrent_kernel, bias):
    """
    Perform matrix multiplication for a single RNN cell.

    Parameters
    ----------
    inputs : 2D tensor with shape (batch_size, input_dim)
    prev_state : 2D tensor with shape (batch_size, units). 
                    The previous output of the RNN.
    prev_carry : 2D tensor with shape (batch_size, units)
    kernel : 2D tensor with shape (input_dim, units)
    recurrent_kernel : 2D tensor with shape (units, units)
    bias : 1D tensor with shape (units,)

    Returns
    -------
    h : Output - 2D tensor with shape (batch_size, units)
    state : List containing 2 2D tensors each with shape (batch_size, units). 
            The first tensor is the output and the second tensor is the carry.
    
    """
    z = np.dot(inputs, kernel)
    z += np.dot(prev_state, recurrent_kernel)
    z = np.add(z, bias)

    z= np.split(z, 4, axis=1);
    c,o=compute_output_and_carry(z,prev_carry);
    h=o*activation(c);
    return h,[h,c];
def compute_output_and_carry(z,prev_carry):
    """
    Compute the output and carry of a single RNN cell.

    Parameters
    ----------
    z : List/tuple containing 4 2D tensors each with shape (batch_size, units).
        The 4 tensors are the result of splitting the output of the matrix
        multiplication of the inputs and the weights.
    prev_carry : 2D tensor with shape (batch_size, units)

    Returns
    -------
    c : 2D tensor with shape (batch_size, units). 
        The carry of the RNN cell.
    o : 2D tensor with shape (batch_size, units). 
        The output of the RNN cell prior to multiplying by activation(c)
    """
    z0, z1, z2, z3 = z
    i = recurrent_activation(z0)
    f = recurrent_activation(z1)
    c = f * prev_carry + i * activation(z2)
    o = recurrent_activation(z3)
    return c, o

def cross_correlation_1d_same(x, kernel):
    """
        Perform cross-correlation operation with 'same' padding. This function is equivalent to
        keras.backend.conv1d(x, kernel, padding='same').
    """
    # compute padding size
    k_size = kernel.shape[0]
    pad_size = (k_size-1) // 2;# pad size that produces similar results as in tensorflow

    # pad the input array
    x_padded = np.pad(x, pad_size, mode='constant')
    
    # initialize output array
    out = np.zeros_like(x)

    # perform convolution
    for i in range(len(out)):
        d=x_padded[i:i+k_size];
        if d.shape[0]<kernel.shape[0]:
            # This is a little messy but needed to produce
            # tensorflow-like result.
            d=np.pad(d, (0, kernel.shape[0]-d.shape[0]), mode='constant');

        out[i] = np.sum(d * kernel)

    return out


def predict_mep(data):
    """
    Predict the MEP of a given sequence.
    """
    # Load the model
    model=keras.models.load_model('mep_classifier.keras');
    # Predict the MEP
    preds = model.predict(data);
    return preds;

def predict_mep_manually(data):
    """
    Predict the MEP of a given sequence by extracting the keras model weights and manually
    performing the matrix multiplications.
    The model has the following architecture:
    Input layer: input_shape=(None, 50, 1), 
    Conv1D layer: 32 filters, kernel_size=12, padding='same', activation='relu'
    LSTM layer: 16 units, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'
    Dense layer: 3 unit, activation='sigmoid'
    """
    
    
    # Load the model
    model=keras.models.load_model('mep_classifier.keras');

    # Get the mean and std of the training data from normalization layer
    mean = model.layers[0].mean.numpy();
    std = model.layers[0].variance.numpy()**0.5;

    # Normalize the data
    data = (data - mean) / std;


    # Extract the weights
    weights = model.get_weights();

    # Conv1D layer: Perform convolution(i.e cross-correlation in signal processing).
    # The input is a 3D tensor with shape (batch_size, steps, input_dim).
    # The output is a 3D tensor with shape (batch_size, new_steps, filters)
    kernels = weights[0];
    biases = weights[1];
    batch_size = data.shape[0];
    if True:
        # Perform without using keras.backend
        data_filtered = np.zeros((batch_size, data.shape[1], kernels.shape[-1]));
        for batch_n in range(batch_size):
            for i in range(kernels.shape[-1]):
                kern = kernels[:, 0, i];
                data_filtered[batch_n,:,i] = cross_correlation_1d_same(data[batch_n,:,0], kern) + biases[i];
        
        data=np.maximum(data_filtered,0); # ReLU activation
        
    else:
        # Perform the same convolution(i.e cross-correlation in signal processing) 
        # operation using keras.backend
        import keras.backend as K;
        tf_filtered=K.conv1d(K.variable(data),K.variable(kernels),padding='same')+biases;
        data=np.maximum(tf_filtered,0);


    # LSTM layer: Perform matrix multiplication.
    # The input is a 3D tensor with shape (batch_size, steps, input_dim).
    # The output is a 2D tensor with shape (batch_size, units)
    # Activation is tanh.
    kernel = weights[2];
    recurrent_kernel = weights[3];
    bias = weights[4];
    steps = data.shape[1];
    batch_size = data.shape[0];
    units=round((kernel.shape[-1])/4);
    prev_state = np.zeros((batch_size, units));# Initial state is zero
    prev_carry=np.zeros((batch_size, units));# Initial carry is zero
    states=[prev_state,prev_carry];  
    output = np.zeros((batch_size, steps, units));
    for step in range(steps):
        output[:, step,:],states = rnn_cell(data[:, step, :], states[0], states[1], kernel, recurrent_kernel, bias);

    
    # Dense layer: Perform matrix multiplication.
    # The input is a 2D tensor with shape (batch_size, units).
    # The output is a 2D tensor with shape (batch_size, units)
    # Activation is sigmoid.
    kernel = weights[5];
    bias = weights[6];
    output = np.dot(output, kernel);
    output = np.add(output, bias);
    output=sigmoid(output);
    return output;


    
def predict_mep_manually_lstm(data):
    """
    Predict the MEP of a given sequence by extracting the keras model weights and manually
    performing the matrix multiplications.
    The model has the following architecture:
    Input layer: input_shape=(None, 50, 1), 
    LSTM layer: 16 units
    Dense layer: 3 unit
    """
    
    # Load the model
    model=keras.models.load_model('mep_classifier.keras');

    # Get the mean and std of the training data from normalization layer
    mean = model.layers[0].mean.numpy();
    std = model.layers[0].variance.numpy()**0.5;

    # Normalize the data
    data = (data - mean) / std;

    # Extract the weights
    weights = model.get_weights();

    

    # LSTM layer: Perform matrix multiplication.
    # The input is a 3D tensor with shape (batch_size, steps, input_dim).
    # The output is a 2D tensor with shape (batch_size, units)
    # Activation is tanh.
    kernel = weights[0];
    recurrent_kernel = weights[1];
    bias = weights[2];
    steps = data.shape[1];
    batch_size = data.shape[0];
    units=round((kernel.shape[-1])/4);
    prev_state = np.zeros((batch_size, units));# Initial state is zero
    prev_carry=np.zeros((batch_size, units));# Initial carry is zero
    states=[prev_state,prev_carry];
    output = np.zeros((batch_size, steps, units));
    for step in range(steps):
        output[:, step,:],states = rnn_cell(data[:, step, :], states[0], states[1], kernel, recurrent_kernel, bias);

    
    
    # Dense layer: Perform matrix multiplication.
    # The input is a 2D tensor with shape (batch_size, units).
    # The output is a 2D tensor with shape (batch_size, units)
    # Activation is sigmoid.
    kernel = weights[3];
    bias = weights[4];
    output = np.dot(output, kernel);
    output = np.add(output, bias);
    output=sigmoid(output);
    return output;
    

    
# Generate a random sequence
data = np.random.randn(2048, 20, 1);


# # Predict the MEP through the keras model
preds = predict_mep(data);
num_non_bg=np.count_nonzero(preds[:,-1]>0.5);
print(f'Using model\'s predict method:\n {preds}');
print(f'Number of non-background predictions:{num_non_bg}')



# Predict the MEP manually
#preds_all = predict_mep_manually_lstm(data);
# preds_all = predict_mep_manually(data);
# preds = preds_all[:,-1,:];
# print(f'Manual prediction:\n{preds}');

def sanity_test():
    """
    Sanity test to check if the manual prediction is correct.
    """
    
    # Use scipy to load expected classifier
    from scipy.io import loadmat;
    classifier = loadmat('mep_classifier.mat');
    sanity_test_inputs = classifier['sanity_test_inputs'];
    sanity_test_outputs = classifier['sanity_test_outputs'];

    actual_preds =  predict_mep_manually(sanity_test_inputs);
    actual_preds = actual_preds[:,-1,:];

    # Check if the predictions are correct
    if np.allclose(actual_preds, sanity_test_outputs):
        print('Sanity test passed');
    else:
        print('Sanity test failed');

sanity_test();






    