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
fnames.append(os.path.join('./data/H1_s1_session1_both_c3-c4_relax_s4_smep-recruitment-c3-c4_labelled.csv'));
fnames.append(os.path.join('./data/H2_s1_session1_both_c3-c4_relax_s1_smep-recruitment-c3-c4_labelled.csv'));
fnames.append(os.path.join('./data/H3_s1_session1_both_c3-c4_relax_s2_smep-recruitment-c3-c4_labelled.csv'));
fnames.append(os.path.join('./data/H4_s1_session1_both_c3-c4_relax_s4_smep-recruitment-c3-c4_labelled.csv'));
fnames.append(os.path.join('./data/H21_s1_session1_both_c3-c4_relax_s2_smep-recruitment-c3-c4_labelled.csv'));
fnames.append(os.path.join('./data/H22_s1_session1_both_c3-c4_relax_s1_smep-recruitment-c3-c4_labelled.csv'));
fnames.append(os.path.join('./data/H23_s1_session1_both_c3-c4_relax_s1_smep-recruitment-c3-c4_labelled.csv'));

test_fnames=[fnames[3]];

# Load data
data,targets=loadResponseData(test_fnames);

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


# Convert to binary classification by combining bacground and stim artefact.
if True:
    for n in range(len(test_targets)):
        if test_targets[n,0]==1 or test_targets[n,1]==1:
            test_targets[n,:]=[0,0,0];
    test_targets=np.sum(test_targets,axis=1);


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











