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
    
    # targets
    targets[start:end,:]=np.array([0,1]);# whole wave



# Plot the data

plt.plot(data[800:1300,:]);
plt.plot(targets[800:1300,:])

# Normalise data
mean=np.mean(data,axis=0);# Using all data for training
std=np.std(data,axis=0);# Using all data for training
train_data=(data-mean)/std;

# Prepare dataset
sampling_rate=1;
sequence_length=20;
batch_size=256;

train_dataset=keras.utils.timeseries_dataset_from_array(
    data=train_data,
    targets=targets,
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=None,
    );

# Create the simple model for a start
inputs=keras.Input(shape=(sequence_length, num_features));
x=keras.layers.LSTM(32,return_sequences=False)(inputs);

x=keras.layers.Dense(18,activation='relu')(x);
outputs=keras.layers.Dense(2,activation='softmax')(x);
model=keras.Model(inputs=inputs, outputs=outputs);

# Compile the model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']);
model.summary();

# Train the model
callbacks=[
    #keras.callbacks.ModelCheckpoint('motor_evoked_potential_minimal_save_at_{epoch}.keras'),
    ];
history=model.fit(
    train_dataset,
    epochs=10,
    callbacks=callbacks,
    );

# plot the loss
plt.figure()
plt.plot(history.history['loss']);
plt.show();









