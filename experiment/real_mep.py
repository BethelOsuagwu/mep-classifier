from tensorflow import keras;
import numpy as np;
import matplotlib.pyplot as plt;
import os;

# Experiment to detect motor evoked potential(EPR).
# The EPR is a response of the muscles to a stimulus.

# data
training_fname=os.path.join('training_data.csv');

with open(training_fname) as f:
    import_data=f.read();
    
data_lines=import_data.split("\n");
num_features=len(data_lines[0].split(","));

data=np.zeros((len(data_lines),num_features));
for i in range(0,len(data_lines)-1):# the last row is empty so -1
    line=data_lines[i].split(',');
    line=[float(d) for d in line];
    data[i,:]=line[:];
    


# targets
targets_fname=os.path.join('targets_data.csv');
with open(targets_fname) as f:
    import_targets=f.read();
target_lines=import_targets.split("\n");
num_targets=len(target_lines[0].split(","));

targets=np.zeros((len(data_lines),num_targets));
for i in range(0,len(data_lines)-1):# the last row is empty so -1
    line=target_lines[i].split(',');
    line=[float(d) for d in line];
    targets[i,:]=line[:];


# Preprocess
# remove baseline
background=[];
for i in range(0,len(data)):
    if targets[i,0]==1:
        background.append(data[i,:]);
background_mean=np.mean(background);
data=data-background_mean;

#
num_samples=len(data);


import pandas as pd;
dataset = np.concatenate((data,targets),axis=1) #raw dataset
dataset = pd.DataFrame(dataset)

from keras import layers
neuron = 16
L = 30 #sequence length, no need to be too big
W =1 # signal channel number
output_num = num_targets
model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(neuron,input_shape=(L,W))))
model.add(layers.Dense(output_num,activation = "sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

def sequence_generator(data,L,W,D,O):# used to generate proper data strcture for RNN training
  sequence_group=[]
  for i in range(len(data)-L-D):
    sequence_group.append(data.iloc[i:i+L+D])  
  data_= np.array([df.values for df in sequence_group])
  input = data_[:,:L,0:W]
  target = data_[:,-1,W:W+O]
  return input,target

inputs,targets = sequence_generator(dataset,L,W,0,output_num) # input and target for training

history = model.fit(inputs,targets,batch_size=2048,epochs=50)
predict = model.predict(inputs)
plt.plot(data[L:1000]) 
plt.plot(predict[:1000,:]) #plot the predicted class
plt.show()









