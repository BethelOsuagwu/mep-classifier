from tensorflow import keras;
import numpy as np;
import matplotlib.pyplot as plt;

# Experiment to detect motor evoked potential(EPR).
# The EPR is a response of the muscles to a stimulus.
# EPRs will be simulated here as sine waves.

# Generate data
num_samples=100000;
num_features=1;
data=np.random.random((num_samples, num_features));
targets=np.ones((num_samples,2));
targets[:,1:]=0;


# Generate data - Insert sine waves in the data as pretend ERPs.
num_sine_waves=1000;
steps=round(num_samples/num_sine_waves);
sine_wave=2*np.sin(np.linspace(0, 2*np.pi, 50));
for i in range(num_sine_waves-1):
    start=(1+i)*steps;
    end=start+sine_wave.shape[0];
    data[start:end,0]=sine_wave[:];
    
    crest_pos=start+round(sine_wave.shape[0]/4.0);
    trough_pos=start+round(3*sine_wave.shape[0]/4.0);
    
    # target of [0,0,0] rep background data
    targets[start:end,:]=np.array([0,1]);# whole wave

import pandas as pd;
dataset = np.concatenate((data,targets),axis=1) #raw dataset
dataset = pd.DataFrame(dataset)

from keras import layers
neuron = 16
L = 20 #sequence length, no need to be too big
W =1 # signal channel number
output_num = 2
model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(neuron,input_shape=(L,W))))
model.add(layers.Dense(output_num,activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
plt.plot(data[L:500]) 
plt.plot(predict[:500,:]) #plot the predicted class
plt.show()









