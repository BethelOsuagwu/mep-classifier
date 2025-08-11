from scipy.io import savemat;
import tensorflow as tf;
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
    getModel,
    getModelSmall,
    export_model
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

val_fnames=[fnames[3]];# Note that except for H2, the validation data here is not used to tune the classifier. The validation data here will be used for testing.
train_fnames=[fn for fn in fnames if fn!=val_fnames[0] ];


# Load data
data,targets=loadResponseData(train_fnames);

# Load validation data
val_data,val_targets=loadResponseData(val_fnames);


print('\n\n% of classes in training data')
countClasses(targets);

print('\n\n% of classes in training data after discarding some')
data,targets=discardExcessBackgroundData(data,targets,discard_percent=0);

print('\n\n% of classes in validation data')
countClasses(val_targets);



    

# Model parameters
sequence_time_len=10;# The length of a sequence in milliseconds
sequence_len = np.round(sequence_time_len/1000 * sample_freq).astype('int');
num_channel =1 # signal channel number
num_classes=1 ;

# Generate data
using_causal_data=False;
if using_causal_data:
    data_seq,targets_seq = sequence_generator(data,targets,sequence_len,0) # input and target for training
    val_data,val_targets=sequence_generator(val_data,val_targets,sequence_len,0);
else:
    data_seq,targets_seq = sequence_generator_non_causal(data,targets,sequence_len) # input and target for training
    val_data,val_targets=sequence_generator_non_causal(val_data,val_targets,sequence_len);
# Split data into training, evaluation and test sets
(train_data,
 train_targets,
 _,
 _,
 _,
 _)=trainValTestSplit(data_seq,targets_seq,train_percent=100,val_percent=0);




# Build model
bg_mean,bg_std=normalisationParameters(train_data[:,0,:],train_targets);
model=getModelSmall(num_classes,bg_mean,bg_std);


# Convert to binary classification by combining bacground and stim artefact.
if True:
    for n in range(len(train_targets)):
        if train_targets[n,0]==1 or train_targets[n,1]==1:
            train_targets[n,:]=[0,0,0];
    train_targets=np.sum(train_targets,axis=1);
    for n in range(len(val_targets)):
        if val_targets[n,0]==1 or val_targets[n,1]==1:
            val_targets[n,:]=[0,0,0];
    val_targets=np.sum(val_targets,axis=1);

# Train model
model_base_filename='uclassifier'
callbacks=[
    keras.callbacks.ModelCheckpoint(
        filepath=model_base_filename+'.keras',
        monitor='val_loss',
        save_best_only=True)
    ];
history = model.fit(
    train_data,
    train_targets,
    batch_size=2048,
    epochs=1*50,
    validation_data=(val_data,val_targets),
    callbacks=callbacks)

# Load the best model
model = keras.models.load_model(model_base_filename+'.keras');

# Save the model to saved odel format
#tf.save_model(model, './model')
keras.models.save_model(model,'./export/model',save_format='tf')

# Export the weights, mean and variance  etc to matlab.
#TODO: convert to a function
#weights = model.get_weights()
#norm_layer=model.get_layer(index=0);
#mean = norm_layer.mean.numpy();
#variance = norm_layer.variance.numpy();
sanity_test_inputs = np.random.randn(3,sequence_len,num_channel);
sanity_test_outputs = model.predict(sanity_test_inputs);
savemat(model_base_filename+'.mat',{
    'sample_freq':sample_freq,
    'layers':export_model(model),
    'sanity_test_inputs':sanity_test_inputs, 
    'sanity_test_outputs':sanity_test_outputs
    });


# KMP_DUPLICATE_LIB_OK
os.environ['KMP_DUPLICATE_LIB_OK']='True';#TODO: this is to fix matplotlib causing tensorflow crash; but that may cause crashes or silently produce incorrect results.

# predict
predict = model.predict(data_seq)
plt.plot(data[sequence_len:500+sequence_len]) 
plt.plot(predict[:500,:]) #plot the predicted class
plt.show()


# plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()

# plot the accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()







