from scipy.io import savemat;
import tensorflow as tf;
from keras import layers
import pandas as pd;
from tensorflow import keras;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
import keras.backend as K;
from custom_layers import Implode;


# Experiment to detect motor evoked potential(EPR).
# The EPR is a response of the muscles to a stimulus.



# Data
sample_freq = 4000;#Hz

training_fnames=[None];
training_fnames[0]=os.path.join('./data/improved-training-09-06-2023.csv');

test_fnames=[None];
test_fnames[0]=os.path.join('./data/testing-18-05-2023.csv');


# Load Training data
data_lines=[];
for fname in training_fnames:
    with open(fname) as f:
        import_data=f.read();
        data_lines_raw=import_data.split("\n");
        data_lines=data_lines+data_lines_raw[1:];# Item 0 is headers

num_features=1;
num_classes=len(data_lines[0].split(",")) - num_features;
num_samples=len(data_lines);

data=np.zeros((num_samples,num_features));
targets=np.zeros((num_samples,num_classes));
for i in range(0,num_samples):
    line=data_lines[i].split(',');
    if len(line) is not num_features+num_classes:
        print(f'Skipping line {i} in data b/c of incorrect format');
        continue;
    feature_line=[float(d) for d in line[0:num_features]];
    data[i,:]=feature_line[:];
    
    target_line=[float(d) for d in line[num_features:]];
    targets[i,:]=target_line[:];


# Load Testing data
data_lines=[];
for fname in test_fnames:
    with open(fname) as f:
        import_data=f.read();
        data_lines_raw=import_data.split("\n");
        data_lines=data_lines+data_lines_raw[1:];# Item 0 is headers


test_data=np.zeros((num_samples,num_features));
test_targets=np.zeros((num_samples,num_classes));
for i in range(0,num_samples):
    line=data_lines[i].split(',');
    if len(line) is not num_features+num_classes:
        print(f'Skipping line {i} in data b/c of incorrect format');
        continue;
    feature_line=[float(d) for d in line[0:num_features]];
    test_data[i,:]=feature_line[:];
    
    target_line=[float(d) for d in line[num_features:]];
    test_targets[i,:]=target_line[:];



# Experiment with converting the stim artifact class into sub class of response
if False:
    for n in range(len(targets)):
        if targets[n,1]==1:
            targets[n,:]=[0,1,1];
    for n in range(len(test_targets)):
        if test_targets[n,1]==1:
            test_targets[n,:]=[0,1,1];



# Discard excess train baseline data to balance the classes
discard_idxs=[];
discard_percent=5;
for n in range(len(targets)):
    if targets[n,0]==1 and (np.random.rand()>((100-discard_percent)/100)):
        discard_idxs.append(n);
data=np.delete(data,discard_idxs,axis=0);
targets=np.delete(targets,discard_idxs,axis=0);


bg_targets_idx=targets[:,0]==1;
bg_percent=np.count_nonzero(bg_targets_idx)/len(targets);# percentage of background class relative to all other classes
ep_percent=(np.count_nonzero(targets[:,2]==1)-np.count_nonzero(np.logical_and(targets[:,1],targets[:,2])))/len(targets);
stim_artefact_percent=(np.count_nonzero(targets[:,1]==1))/len(targets);
print (f'Train: Percentage of background class: {bg_percent}')
print (f'Train: Percentage of EP class: {ep_percent}')
print (f'Train: Percentage of stim artefact class: {stim_artefact_percent}')

# Discard excess test baseline data to balance the classes
discard_idxs=[];
for n in range(len(test_targets)):
    if test_targets[n,0]==1 and (np.random.rand()>((100-discard_percent)/100)):
        discard_idxs.append(n);
test_data=np.delete(test_data,discard_idxs,axis=0);
test_targets=np.delete(test_targets,discard_idxs,axis=0);

# Discard all test artefact data as they are the wrong artefact
discard_idxs=[];
for n in range(len(test_targets)):
    if test_targets[n,1]==1:
        discard_idxs.append(n);
test_data=np.delete(test_data,discard_idxs,axis=0);
test_targets=np.delete(test_targets,discard_idxs,axis=0);

bg_targets_idx=test_targets[:,0]==1;
bg_percent=np.count_nonzero(bg_targets_idx)/len(test_targets);# percentage of background class relative to all other classes
ep_percent=(np.count_nonzero(test_targets[:,2]==1)-np.count_nonzero(np.logical_and(test_targets[:,1],test_targets[:,2])))/len(test_targets);
stim_artefact_percent=(np.count_nonzero(test_targets[:,1]==1))/len(test_targets);
print (f'Test: Percentage of background class: {bg_percent}')
print (f'Test: Percentage of EP class: {ep_percent}')
print (f'Test: Percentage of stim artefact class: {stim_artefact_percent}')

# Experiment with removing the artefact class
if False:
    discards=[];
    for n in range(len(targets)):
        if targets[n,1]==1:
            discards.append(n);
    targets=np.delete(targets,discards,axis=0);
    data=np.delete(data,discards,axis=0)
    targets[:,1]=targets[:,2];
    targets=targets[:,0:2]
    num_classes=2;
    
    discards=[];
    for n in range(len(test_targets)):
        if test_targets[n,1]==1:
            discards.append(n);
    test_targets=np.delete(test_targets,discards,axis=0);
    data=np.delete(data,discards,axis=0)
    test_targets[:,1]=test_targets[:,2];
    test_targets=test_targets[:,0:2]
    num_classes=2;
    print('Artefact class is removed')


# Split train data into training, evaluation
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

# Testing dataset
test_dataset=np.concatenate((test_data,test_targets),axis=1);
test_dataset=pd.DataFrame(test_dataset);

# Model parameters
neuron = 16
sequence_time_len=10;# The length of a sequence in milliseconds
L = np.round(sequence_time_len/1000 * sample_freq).astype('int'); #sequence length, no need to be too big
W =1 # signal channel number
output_num = num_classes



# Model
model = keras.Sequential()

## Normalisation layer
model.add(layers.Normalization(axis=-1,mean=bg_mean,variance=bg_std**2));

## Spatial
# =============================================================================
# model.add(layers.Conv1D(32, 8,activation='relu',padding='same'));
# model.add(layers.MaxPooling1D(pool_size=2));
# model.add(layers.Conv1D(64, 8,activation='relu',padding='same'));
# model.add(layers.MaxPooling1D(pool_size=2));
# model.add(layers.Conv1D(128, 8,activation='relu',padding='same'));
# model.add(layers.MaxPooling1D(pool_size=2));
# model.add(layers.Conv1D(256, 8,activation='relu',padding='same'));
# model.add(layers.MaxPooling1D(pool_size=2));
# =============================================================================
model.add(layers.Conv1D(32, 8,activation='relu',padding='same'));
model.add(layers.MaxPooling1D(pool_size=2));
model.add(layers.Conv1D(64, 8,activation='relu',padding='same'));
model.add(layers.MaxPooling1D(pool_size=2));
model.add(layers.Conv1D(128, 8,activation='relu',padding='same'));
model.add(layers.MaxPooling1D(pool_size=2));
model.add(layers.Conv1D(256, 8,activation='relu',padding='same'));
model.add(layers.UpSampling1D(size=2));
model.add(layers.Conv1DTranspose(128, 2,activation='relu'));
model.add(layers.UpSampling1D(size=2));
model.add(layers.Conv1DTranspose(64, 8,activation='relu'))
model.add(layers.Conv1DTranspose(1, 12,activation='relu'))



#model.add(Implode())

## Temporal
model.add(layers.LSTM(neuron));

## Output
model.add(layers.Dense(output_num,activation = "sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Training
def sequence_generator(data,L,W,D,O):# used to generate proper data strcture for RNN training
  # Generate Non-causal input RNN data
  sequence_group=[]
  for i in range(len(data)-L-D):
    sequence_group.append(data.iloc[i:i+L+D])  
  data_= np.array([df.values for df in sequence_group])
  input = data_[:,:L,0:W]
  target = data_[:,-1,W:W+O]
  return input,target

def sequence_generator_non_causal(data,L,nchan,unused,nclasses):# used to generate proper data strcture for RNN training
  # Generate Non-causal input RNN data
  sequence_group=[]
  for i in range(len(data)-(L-1)):
    sequence_group.append(data.iloc[i:i+L])  
  data_= np.array([df.values for df in sequence_group])
  inputs = data_[:,:L,0:nchan]
  target = data_[:,0,nchan:nchan+nclasses]
  return inputs,target

def test_sequence_generator_non_causal():
    a=np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6]])
    expected_inputs=np.array([[
            [1],
            [2],
            [3]],

           [[2],
            [3],
            [4]],

           [[3],
            [4],
            [5]],
           
           [[4],
            [5],
            [6]]]);
    expected_targets=np.array([
            [1],
            [2],
            [3],
            [4]]);
    actual_inputs,actual_targets=sequence_generator_non_causal(pd.DataFrame(a), 3, 1,0, 1);
    assert np.all(expected_inputs==actual_inputs),f"Expected inputs is:\n {expected_inputs} \n but got\n {actual_inputs}";
    assert np.all(expected_targets==actual_targets),f"Expected targets is:\n {expected_targets} \n but got\n {actual_targets}";
    print('Pass')
    
    
using_causal_data=False;
if using_causal_data:
    inputs,targets = sequence_generator(train_dataset,L,W,0,output_num) # input and target for training
    val_data,val_targets = sequence_generator(val_dataset,L,W,0,output_num) # input and target for validation
    test_data_input, test_data_targets=sequence_generator(test_dataset,L,W,0,output_num)
else:
    inputs,targets = sequence_generator_non_causal(train_dataset,L,W,0,output_num) # input and target for training
    val_data,val_targets = sequence_generator_non_causal(val_dataset,L,W,0,output_num) # input and target for validation
    test_data_input, test_data_targets=sequence_generator_non_causal(test_dataset,L,W,0,output_num)

# Print summary
if False:
    model.build((None,L,1))
    model.summary()
    exit()

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


# KMP_DUPLICATE_LIB_OK
os.environ['KMP_DUPLICATE_LIB_OK']='True';#TODO: this is to fix matplotlib causing tensorflow crash; but that may cause crashes or silently produce incorrect results.

# predict
predict = model.predict(inputs)
plt.plot(train_data[L:500+L]) 
plt.plot(predict[:500,:]) #plot the predicted class
plt.show()

# Evaluate the classifier
test_result=model.evaluate(
    test_data_input, test_data_targets
    );
print(f'Test loss:{test_result[0]}, Test accuracy:{test_result[1]}')

# Save to visualize in matlab
savemat('traindata_predicted.mat',{'train_data':train_data,'predict':predict,'L':L})

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

# Visualise the output of the conv1D layers
visual_model=keras.Sequential();
for layer in model.layers[:-2]:# Exclude the last two layers
    visual_model.add(layer);

visual_model.build((None,L,1));
visual_model.summary();


conv_outputs=visual_model.predict(inputs);

# visualise the visual model output
offset=-6000;
start_idx=9000+offset;
stop_idx=10000+offset;
input_scale=1000;
target_scale=1000;
sumed_output = np.sum(conv_outputs,axis=(1));

# Adjust train data/targets depending on the sequence generator used
adj_train_data=train_data;
adj_train_targets=train_targets;
if using_causal_data:
    adj_train_data=train_data[L:];
    adj_train_targets=train_targets[L:];

#
for chan in range(1):
    plt.plot(sumed_output[start_idx:stop_idx,chan],label=f'output channel {chan}')
plt.plot(adj_train_data[start_idx:stop_idx,0]*input_scale,label='inputs')
plt.plot(adj_train_targets[start_idx:stop_idx,0]*target_scale,label='Background')
plt.plot(adj_train_targets[start_idx:stop_idx,1]*target_scale,label='Artefact')
plt.plot(adj_train_targets[start_idx:stop_idx,2]*target_scale,label='Response')
#plt.plot(conv_outputs[:,1])
#plt.plot(conv_outputs[:,2])
#plt.plot(conv_outputs[:,3])
plt.title(f'Conv layer input(x {input_scale}) & output for channel {chan}')
plt.ylabel('amplitude')
plt.xlabel('samples')
plt.legend( loc='upper left')
plt.show()







