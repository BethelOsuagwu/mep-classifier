from tensorflow import keras;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
import keras.backend as K;

# Experiment to detect motor evoked potential(EPR).
# The EPR is a response of the muscles to a stimulus.

# data
training_fname=os.path.join('training_data_with_artefact_class_all_response_win.csv');

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

bg_tragets_idx=targets[:,0]==1;
bg_percent=np.count_nonzero(bg_tragets_idx)/len(targets);# percentage of background clas relative to all other classes
ep_percent=(np.count_nonzero(targets[:,2]==1)-np.count_nonzero(targets[:,1]==1))/len(targets);
stim_artefact_percent=(np.count_nonzero(targets[:,1]==1))/len(targets);
print (f'Percentage of background class: {bg_percent}')
print (f'Percentage of EP class: {ep_percent}')
print (f'Percentage of stim artefact class: {stim_artefact_percent}')


# Preprocess
# remove baseline
# =============================================================================
# background=[];
# for i in range(0,len(data)):
#     if targets[i,0]==1:
#         background.append(data[i,:]);
# background_mean=np.mean(background);
# data=data-background_mean;
# =============================================================================

# Get all background data for training data.
# Calculate the mean and std.
# Use the mean and std to comput the z-score of all data
bg_idx=targets[:,0]==1;
bg_mean=np.mean(data[bg_idx]);
bg_std=np.std(data[bg_idx]);
data=(data-bg_mean)/bg_std;


#
num_samples=len(data);


import pandas as pd;
dataset = np.concatenate((data,targets),axis=1) #raw dataset
dataset = pd.DataFrame(dataset)

from keras import layers
neuron = 16
L = 50 #sequence length, no need to be too big
W =1 # signal channel number
output_num = num_classes

# Classes
import tensorflow as tf;
class Peak2Peak(layers.Layer):
    def __init__(self,units=1,activation=None):
        """
        Scales each input feature with peak to peak value of each feature.

        Parameters
        ----------
        units : integer, optional
            Units greater than 1 is not implemented. The default 
            is 1.
        activation : function Keras activation, optional
            The default is None.

        Returns
        -------
        TF tensor with same shape as input
        
        Call parameter
        inputs : Tensor
            Tensor of shape (batch_size,time_steps,features)

        """
        super().__init__(dynamic=False,name='peak2peak_layer');
        self.units=units;
        self.activation=activation;
    def build(self,input_shape):
        num_features=input_shape[-1];
        self.W=self.add_weight(shape=(num_features,2),
                               initializer="random_normal",
                               name='peak2peak_weight',
                               trainable=True);
        self.b=self.add_weight(shape=(self.units,2),
                               initializer="zeros",
                               trainable=True);
    def call(self,inputs):
        
        #peak2peak=tf.subtract(inputs[:,0,:], inputs[:,-1,:]);#peak2peak per feature
        #peak2peak=tf.abs(peak2peak);
        #x=tf.multiply(self.W,peak2peak) +self.b;
                      
        peak1=inputs[:,0,:];
        peak2=inputs[:,-1,:];
        peaks=tf.concat([peak1,peak2],axis=1)
        peaks=tf.multiply(peaks,self.W) +self.b;
        peak2peak=tf.subtract(peaks[:,0],peaks[:,1]);
        x=tf.abs(peak2peak);
        
        
        batch_size = K.shape(inputs)[0];
        F=inputs.shape[-1];# number features
        
        
        x=tf.reshape(x,(batch_size,1,F));
        y=tf.multiply(inputs,x);
        
        if self.activation is not None:
            y=self.activation(y);
        return y;
    
# peak2peak_layer=Peak2Peak(1);
# p2p_output=peak2peak_layer(tf.ones(shape=(1,2,7)))
# print(p2p_output.shape)
# exit()

# Spatial => temporal
model = keras.Sequential()
#model.add(Peak2Peak(1))
model.add(layers.Conv1D(16, 12,activation='relu',padding='same'));
#model.add(layers.Conv1D(32, 12,activation='relu',padding='same'));
#model.add(layers.GlobalAveragePooling1D(data_format='channels_last'))
model.add(layers.Bidirectional(layers.LSTM(neuron)));# TODO  try with bidirectional LSTM and also the ConvLSTM
#model.add(layers.Flatten())
model.add(layers.Dense(output_num,activation = "sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
#model.build(input_shape=(None,L,W))
#model.summary();
#exit()
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
plt.plot(data[L:500]*bg_std+bg_mean) 
plt.plot(predict[:500,:]) #plot the predicted class
plt.show()

# plot the loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# plot the accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()









