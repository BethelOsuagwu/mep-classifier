from scipy.io import savemat;
import tensorflow as tf;
from tensorflow import keras;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
import math;
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
fnames.append(os.path.join('./data/H1_mdt.csv'));
fnames.append(os.path.join('./data/H2_mdt.csv'));
fnames.append(os.path.join('./data/H3_mdt.csv'));
fnames.append(os.path.join('./data/H4_mdt.csv'));
fnames.append(os.path.join('./data/H21_mdt.csv'));
fnames.append(os.path.join('./data/H22_mdt.csv'));
fnames.append(os.path.join('./data/H23_mdt.csv'));

test_fnames=[fnames[3]];
val_fnames=[fnames[1]];# Note that except for H2, the validation data here is not used to tune the classifier. The validation data here will be used for testing.
train_fnames=[fn for fn in fnames if (fn!=val_fnames[0] and fn!=test_fnames[0]) ];

num_features=3;

# Load data
data,targets=loadResponseData(train_fnames,num_features,3);

# Load validation data
val_data,val_targets=loadResponseData(val_fnames,num_features,3);


print('\n\n% of classes in training data')
countClasses(targets);

print('\n\n% of classes in training data after discarding some')
data,targets=discardExcessBackgroundData(data,targets,discard_percent=0);


print('\n\n% of classes in validation data')
countClasses(val_targets);

# Model parameters
sequence_time_len=10;# The length of a sequence in milliseconds
sequence_len = np.round(sequence_time_len/1000 * sample_freq).astype('int');
#num_channel =num_features # signal channel number
num_classes=targets.shape[1];

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
 _,_,_,_)=trainValTestSplit(data_seq,targets_seq,train_percent=100,val_percent=0);


# Build model
bg_mean,bg_std=normalisationParameters(train_data[:,0,:],train_targets);# Note that we only need to take one sample from each sequence to get back the unsequenced data given that sequence overlap is= sequence_len-1. Although trainValTestSplit may have scrambled the sequences but we do not care about order to compute mean and std.

model=getModel(num_classes,bg_mean,bg_std);

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
    epochs=30,
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
sanity_test_inputs = np.random.randn(3,sequence_len,num_features);
sanity_test_outputs = model.predict(sanity_test_inputs);
savemat(model_base_filename+'.mat',{
    'input_shape':np.concatenate(([[math.nan],model.input_shape[-2:]])),
    'output_shape':np.array([math.nan,model.output_shape[-1]]),
    'sample_freq':sample_freq,
    'layers':export_model(model),
    'sanity_test_inputs':sanity_test_inputs, 
    'sanity_test_outputs':sanity_test_outputs
    });


# KMP_DUPLICATE_LIB_OK
os.environ['KMP_DUPLICATE_LIB_OK']='True';#TODO: this is to fix matplotlib causing tensorflow crash; but that may cause crashes or silently produce incorrect results.

# predict
predict = model.predict(data_seq)
if using_causal_data:
    plt.plot(data[sequence_len:500+sequence_len],label="data") 
else:
    plt.plot(data[:500],label="data") 
plt.plot(predict[:500,0],label="Bg") #plot the predicted class
plt.plot(predict[:500,1],label="stim artefact") #plot the predicted class
plt.plot(predict[:500,2],label="response") #plot the predicted class
#plt.legend({'Data','background','stim artefact','response','ll'})
plt.legend()
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







