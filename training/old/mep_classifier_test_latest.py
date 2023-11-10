# Test the classifier with a dedicated test set

from scipy.io import savemat;
from tensorflow import keras;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
from helpers import (
    loadResponseData,
    discardExcessBackgroundData,
    countClasses,
    sequence_generator,
    sequence_generator_non_causal,
    trainValTestSplit,
    normalisationParameters,
    getModel
)


# Experiment to detect motor evoked potential(EPR).
# The EPR is a response of the muscles to a stimulus.

# Data
sample_freq = 4000;#Hz

fnames=[];
fnames.append(os.path.join('./data/H2_s1_session1_both_c3-c4_relax_s1_smep-recruitment-c3-c4_labelled.csv'));
#fnames.append(os.path.join('./data/H3_s1_session1_both_c3-c4_relax_s2_smep-recruitment-c3-c4_labelled.csv'));
# Load data
data,targets=loadResponseData(fnames);

print('\n\n% of classes in testing data')
countClasses(targets);

print('\n\n% of classes in testing data after discarding some bg data')
data,targets=discardExcessBackgroundData(data,targets,discard_percent=0);



# Model parameters
sequence_time_len=10;# The length of a sequence in milliseconds
sequence_len = np.round(sequence_time_len/1000 * sample_freq).astype('int');
num_channel =1 # signal channel number
num_classes=targets.shape[1];

# Generate data
using_causal_data=False;
if using_causal_data:
    test_data,test_targets = sequence_generator(data,targets,sequence_len,0) 
else:
    test_data,test_targets = sequence_generator_non_causal(data,targets,sequence_len) 



# Load the best model
model = keras.models.load_model('uclassifier.keras');



# KMP_DUPLICATE_LIB_OK
os.environ['KMP_DUPLICATE_LIB_OK']='True';#TODO: this is to fix matplotlib causing tensorflow crash; but that may cause crashes or silently produce incorrect results.

# predict
predict = model.predict(test_data)
plt.plot(data[sequence_len:500+sequence_len]) 
plt.plot(predict[:500,:]) #plot the predicted class
plt.show()

# Evaluate the classifier
test_result=model.evaluate(test_data, test_targets);
print(f'Test loss:{test_result[0]}, Test accuracy:{test_result[1]}')

# Save to visualize in matlab
savemat('traindata_predicted.mat',{'data':data,'predict':predict,'sequence_len':sequence_len})



# Visualise the output of the conv1D layers
# TODO convert to a functiion
visual_model=keras.Sequential();
for layer in model.layers[:-2]:# Exclude the last two layers
    visual_model.add(layer);

visual_model.build((None,sequence_len,1));
visual_model.summary();


conv_outputs=visual_model.predict(test_data);

# visualise the visual model output
offset=6000;
start_idx=9000+offset;
stop_idx=10000+offset;
input_scale=10000;
target_scale=10000;
summed_output = np.sum(conv_outputs,axis=(1));

# Adjust train data/targets depending on the sequence generator used
adj_data=data;
adj_targets=targets;
if using_causal_data:
    adj_data=data[sequence_len:];
    adj_targets=targets[sequence_len:];

#
for chan in range(1):
    plt.plot(summed_output[start_idx:stop_idx,chan],label=f'output channel {chan}')
plt.plot(adj_data[start_idx:stop_idx,0]*input_scale,label='inputs')
plt.plot(adj_targets[start_idx:stop_idx,0]*target_scale,label='Background')
plt.plot(adj_targets[start_idx:stop_idx,1]*target_scale,label='Artefact')
plt.plot(adj_targets[start_idx:stop_idx,2]*target_scale,label='Response')
#plt.plot(conv_outputs[:,1])
#plt.plot(conv_outputs[:,2])
#plt.plot(conv_outputs[:,3])
plt.title(f'Conv layer input(x {input_scale}) & output for channel {chan}')
plt.ylabel('amplitude')
plt.xlabel('samples')
plt.legend( loc='upper left')
plt.show()







