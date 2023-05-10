from scipy.io import savemat;
import tensorflow as tf;
from keras import layers
import pandas as pd;
from tensorflow import keras;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
import keras.backend as K;


# Experiment to detect motor evoked potential(EPR).
# The EPR is a response of the muscles to a stimulus.

# data
training_fname=os.path.join('./data/training_data_with_artefact_class_all_response_win.csv');

with open(training_fname) as f:
    import_data=f.read();
    
data_lines=import_data.split("\n");
num_features=1;
num_classes=len(data_lines[0].split(",")) - num_features;
data_lines=data_lines[1:]
num_samples=len(data_lines);

data=np.zeros((num_samples,num_features));
targets=np.zeros((num_samples,num_classes));
for i in range(0,num_samples-1):# the last row is empty so -1
    line=data_lines[i].split(',');
    feature_line=[float(d) for d in line[0:num_features]];
    data[i,:]=feature_line[:];
    
    target_line=[float(d) for d in line[num_features:]];
    targets[i,:]=target_line[:];
    

# Experiment with converting the stim artifact class into sub class of response
for n in range(len(targets)):
    if targets[n,1]==1:
        targets[n,:]=[0,1,1];
    


# Discard excess baseline data to balance the classes
discard_idxs=[];
discard_percent=91;
for n in range(len(targets)):
    if targets[n,0]==1 and (np.random.rand()>((100-discard_percent)/100)):
        discard_idxs.append(n);
data=np.delete(data,discard_idxs,axis=0);
targets=np.delete(targets,discard_idxs,axis=0);

bg_targets_idx=targets[:,0]==1;
bg_percent=np.count_nonzero(bg_targets_idx)/len(targets);# percentage of background class relative to all other classes
ep_percent=(np.count_nonzero(targets[:,2]==1)-np.count_nonzero(targets[:,1]==1))/len(targets);
stim_artefact_percent=(np.count_nonzero(targets[:,1]==1))/len(targets);
print (f'Percentage of background class: {bg_percent}')
print (f'Percentage of EP class: {ep_percent}')
print (f'Percentage of stim artefact class: {stim_artefact_percent}')


# Split data into training, evaluation
train_data_percent=80;
idx=round(train_data_percent/100 * len(data));
train_data=data[:idx];
train_targets=targets[:idx];
val_data=data[idx:];
val_targets=targets[idx:];

# Preprocess
# Get all background data for training data. So this is is a special 
# normalisation where only background data is used to compute the normalisation 
# data.
# Calculate the mean and std.
# Use the mean and std to compute the z-score of all data
bg_idx=train_targets[:,0]==1;
bg_mean=np.mean(train_data[bg_idx]);
bg_std=np.std(train_data[bg_idx]);


# Training dataset
train_dataset = np.concatenate((train_data,train_targets),axis=1);
train_dataset = pd.DataFrame(train_dataset);

# Validation dataset
val_dataset = np.concatenate((val_data,val_targets),axis=1);
val_dataset = pd.DataFrame(val_dataset);

# Model parameters
neuron = 16
L = 20 #sequence length, no need to be too big
W =1 # signal channel number
output_num = num_classes



# Model
model = keras.Sequential()

## Normalisation layer
model.add(layers.Normalization(axis=-1,mean=bg_mean,variance=bg_std**2));

## Spatial
model.add(layers.Conv1D(32, 12,activation='relu',padding='same'));

## Temporal
model.add(layers.LSTM(neuron));

## Output
model.add(layers.Dense(output_num,activation = "sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Training
def sequence_generator(data,L,W,D,O):# used to generate proper data strcture for RNN training
  sequence_group=[]
  for i in range(len(data)-L-D):
    sequence_group.append(data.iloc[i:i+L+D])  
  data_= np.array([df.values for df in sequence_group])
  input = data_[:,:L,0:W]
  target = data_[:,-1,W:W+O]
  return input,target

inputs,targets = sequence_generator(train_dataset,L,W,0,output_num) # input and target for training
val_data,val_targets = sequence_generator(val_dataset,L,W,0,output_num) # input and target for validation

callbacks=[
    keras.callbacks.ModelCheckpoint(
        filepath='mep_classifier.keras',
        monitor='val_loss',
        save_best_only=True)
    ];
history = model.fit(
    inputs,
    targets,
    batch_size=2048,
    epochs=50,
    validation_data=(val_data,val_targets),
    callbacks=callbacks)

# Load the best model
model = keras.models.load_model('mep_classifier.keras');

# Export the weights, mean and variance  etc to matlab.
sample_freq = 4000;
weights = model.get_weights()
norm_layer=model.get_layer(index=0);
mean = norm_layer.mean.numpy();
variance = norm_layer.variance.numpy();
sanity_test_inputs = np.random.randn(3,L,W);
sanity_test_outputs = model.predict(sanity_test_inputs);
savemat('mep_classifier.mat',{
    'sample_freq':sample_freq,
    'weights':weights,
    'mean':mean,
    'variance':variance,
    'sanity_test_inputs':sanity_test_inputs, 
    'sanity_test_outputs':sanity_test_outputs
    });


# predict
predict = model.predict(inputs)
plt.plot(train_data[L:500+L]) 
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









