# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:04:54 2023

@author: Bethel
"""
from keras import layers
import pandas as pd;
from tensorflow import keras;
import numpy as np;
from typing import List;
from typing import Tuple



def loadResponseData(filenames: List[str],num_features: int=1,num_targets: int=None)->Tuple[np.ndarray, np.ndarray]:
    """
    Load response data

    Parameters
    ----------
    filenames : List[str]
        List file names.
    num_features : int
        The number oof features. The default is 1.
    num_targets : int
        The number of targets. The default is total columns minus num_features.

    Returns
    -------
    Tuple[np.array,np.array]. (data,targets)  The data and targets.
        data: np.array
            Data. Shape: (num_samples,num_features)
        targets: np.array
            Targets. Shape: (num_samples,num_classes)

    """
    data_lines=[];
    for fname in filenames:
        with open(fname) as f:
            import_data=f.read();
            data_lines_raw=import_data.split("\n");
            data_lines=data_lines+data_lines_raw[1:];# Item 0 is headers

    num_classes=num_targets;
    if num_classes is None:
        num_classes=len(data_lines[0].split(",")) - num_features;
        
    num_samples=len(data_lines);
    
    data=np.zeros((num_samples,num_features));
    targets=np.zeros((num_samples,num_classes));
    for i in range(0,num_samples):
        line=data_lines[i].split(',');
        if len(line) < num_features+num_classes:
            print(f'Skipping line {i} with the following content in data b/c of incorrect format');
            print(line)
            continue;
        feature_line=[float(d) for d in line[0:num_features]];
        data[i,:]=feature_line[:];
        
        target_line=[float(d) for d in line[num_features:num_features+num_classes]];
        targets[i,:]=target_line[:];
        
    return data,targets;


def discardExcessBackgroundData(data: np.ndarray,targets: np.ndarray,discard_percent:float=5)->Tuple[np.ndarray, np.ndarray]:
    """
    Discard excess background data.

    Parameters
    ----------
    data : np.ndarray of shape (num_samples,num_features)
        Data.
    targets : np.ndarray of shape (num_samples,num_classes)
        Targets. 
    discard_percent : float, optional 
        Percentage of background data to discard. The default is 5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]. (data,targets)  The data and targets with excess background data removed.
        data: np.array
            Data. Shape: (num_samples,num_features)
        targets: np.array
            Targets. Shape: (num_samples,num_classes)
    """
    discard_idxs=[];
    
    for n in range(len(targets)):
        if targets[n,0]==1 and (np.random.rand()>((100-discard_percent)/100)):
            discard_idxs.append(n);
    data=np.delete(data,discard_idxs,axis=0);
    targets=np.delete(targets,discard_idxs,axis=0);
    
    countClasses(targets);
    return (data,targets)

def countClasses(targets: np.ndarray)->None:
    """
    Count the number of samples in each class and print the results.

    Parameters
    ----------
    targets : np.ndarray of shape (num_samples,num_classes)
        Targets.

    Returns
    -------
    None.

    """
    bg_targets_idx=targets[:,0]==1;
    n_bg=np.count_nonzero(bg_targets_idx);
    n_ep=(np.count_nonzero(targets[:,2]==1)-np.count_nonzero(np.logical_and(targets[:,1],targets[:,2])));
    n_stim_artefact=(np.count_nonzero(targets[:,1]==1));
    
    bg_percent=n_bg/len(targets);# percentage of background class relative to all other classes
    ep_percent=n_ep/len(targets);
    stim_artefact_percent=n_stim_artefact/len(targets);
    print ('____________')
    print (f'Train: Percentage of background class: {bg_percent} (total samples:{n_bg})')
    print (f'Train: Percentage of Response class: {ep_percent} (total samples:{n_ep})')
    print (f'Train: Percentage of stim artefact class: {stim_artefact_percent} (total samples:{n_stim_artefact})')
    print ('------------')


def trainValTestSplit(data: np.ndarray,targets: np.ndarray,train_percent:float=80,val_percent:float=10,shuffle:bool=True)->Tuple[np.ndarray, np.ndarray]:
    """
    Split data into train, validation and test datasets.

    Parameters
    ----------
    data : np.array
        Data. Shape: (num_samples,num_features)
    targets : np.array
        Targets. Shape: (num_samples,num_classes)
    train_percent : float, optional
        Percentage of data to use for training. The default is 80.
    val_percent : float, optional
        Percentage of data to use for validation. The default is 10.
    shuffle : bool, optional
        Shuffle data. The default is True.

    Returns
    -------
    Tuple[np.ndarray]. (train_data,train_targets,val_data,val_targets,test_data,test_targets)
        train_data: np.array
            Data. Shape: (num_samples,num_features)
        train_targets: np.array
            Targets. Shape: (num_samples,num_classes)
        val_data: np.array
            Data. Shape: (num_samples,num_features)
        val_targets: np.array
            Targets. Shape: (num_samples,num_classes)
        test_data: np.array
            Data. Shape: (num_samples,num_features)
        test_targets: np.array
            Targets. Shape: (num_samples,num_classes)

    """

    if train_percent+val_percent>100:
        raise ValueError("train_percent + val_percent must not be greater than 100");

    if shuffle:
        # Create an index array to shuffle the rows
        indices = np.arange(data.shape[0]);
        np.random.shuffle(indices);

        # Shuffle the rows of both arrays using the index array
        data = data[indices];
        targets = targets[indices];

    # Train data
    train_len=round(train_percent/100 * len(data));
    train_data=data[:train_len];
    train_targets=targets[:train_len];

    # Validation data
    val_len=round(val_percent/100 * len(data));
    val_data=data[train_len:train_len+val_len];
    val_targets=targets[train_len:train_len+val_len];

    # Test data
    test_data=[];
    test_targets=[];
    if train_len+val_len<len(data):
        test_data=data[train_len+val_len:];
        test_targets=targets[train_len+val_len:];

    return (train_data,train_targets,val_data,val_targets,test_data,test_targets);

def normalisationParameters(train_data: np.ndarray,train_targets: np.ndarray)->Tuple[float,float]:
    """
    Get normalisation parameters from data.
    Get all background data for training data. So this is is a special 
    normalisation where only background data is used to compute the normalisation 
    data.
    Parameters
    ----------
    train_data : np.ndarray
        Data.
    train_targets : np.ndarray
        Targets.

    Returns
    -------
    Tuple[float,float]. (mean,std)

    """
    bg_idx=train_targets[:,0]==1;
    bg_mean=np.mean(train_data[bg_idx]);
    bg_std=np.std(train_data[bg_idx]);
    return (bg_mean,bg_std);


def getModel(num_classes:int=3,normalisation_mean:float=None,normalisation_std:float=None)->keras.Sequential:
    """
    Get the default model.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. The default is 3.
    normalisation_mean : float, optional
        Mean of normalisation. The default is None.
    normalisation_std : float, optional
        Standard deviation of normalisation. The default is None.

    Returns
    -------
    keras.Sequential
        Model.

    """


    # Model parameters
    output_num = num_classes

    # Model
    model = keras.Sequential()

    ## Normalisation layer
    if normalisation_mean is not None and normalisation_std is not None:
        model.add(layers.Normalization(axis=-1,mean=normalisation_mean,variance=normalisation_std**2));
    
    ## Spatial
    model.add(layers.Conv1D(8, 3,activation='relu',padding='valid'));
    model.add(layers.MaxPooling1D(pool_size=2,strides=2));
    model.add(layers.Conv1D(16, 3,activation='relu',padding='valid'));
    model.add(layers.MaxPooling1D(pool_size=2,strides=2));
    model.add(layers.Conv1D(32, 3,activation='relu',padding='valid'));
    model.add(layers.MaxPooling1D(pool_size=2,strides=2));
    model.add(layers.Conv1D(64, 3,activation='relu',padding='valid'));
    #model.add(layers.UpSampling1D(size=2));
    model.add(layers.Conv1DTranspose(32, 7,activation='relu'));
    #model.add(layers.UpSampling1D(size=2));
    model.add(layers.Conv1DTranspose(8, 5,activation='relu'))
    #model.add(layers.UpSampling1D(size=2));
    model.add(layers.Conv1DTranspose(4, 3,activation='relu'))
    #model.add(layers.UpSampling1D(size=2));
    model.add(layers.Conv1DTranspose(1, 3,activation='relu'))
    #model.add(layers.Dropout(0.5))
    
    # model.add(layers.Conv1D(32, 8,activation='relu',padding='same'));
    # model.add(layers.MaxPooling1D(pool_size=2,strides=2));
    # model.add(layers.Conv1D(64, 8,activation='relu',padding='same'));
    # model.add(layers.MaxPooling1D(pool_size=2,strides=2));
    # model.add(layers.Conv1D(128, 8,activation='relu',padding='same'));
    # model.add(layers.MaxPooling1D(pool_size=2,strides=2));
    # model.add(layers.Conv1D(256, 8,activation='relu',padding='same'));
    # model.add(layers.UpSampling1D(size=2));
    # model.add(layers.Conv1DTranspose(128, 2,activation='relu'));
    # model.add(layers.UpSampling1D(size=2));
    # model.add(layers.Conv1DTranspose(64, 8,activation='relu'))
    # model.add(layers.Conv1DTranspose(32, 12,activation='relu'))


    ## Temporal
    model.add(layers.LSTM(16));#todo try 4 and train for longer

    ## Output
    model.add(layers.Dense(output_num,activation = "sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model;


def getModelSmall(num_classes:int=3,normalisation_mean:float=None,normalisation_std:float=None)->keras.Sequential:
    
    # Model parameters
    output_num = num_classes
    
    model = keras.Sequential()
    ## Normalisation layer
    if normalisation_mean is not None and normalisation_std is not None:
        model.add(layers.Normalization(axis=-1,mean=normalisation_mean,variance=normalisation_std**2));
    
    model.add(layers.Conv1D(32, 12,activation='relu',padding='same'));
    
    model.add(layers.LSTM(16));
    
    model.add(layers.Dense(output_num,activation = "sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model;

def export_model(model:keras.Model)->List[dict]:
    """
    Exports the layer weights and config a list of dictionary in the order they 
    are contained in the model.

    Parameters
    ----------
    model : keras.Model
        Keras model.

    Returns
    -------
    List[dict]
        Exported model layers where each item is a dictionary with keys {weights,config}.

    """
    layers=[];
    
    

    for layer in model.layers:
        w=layer.get_weights();
        c=dict_replace_none_with_empty_array(layer.get_config());
        layers.append({'weights':w,'config':c,'type':type(layer).__name__});

    return layers;

# Training
def sequence_generator(data:np.ndarray,targets:np.ndarray,L:int,D:int)->Tuple[np.ndarray,np.ndarray]:
    """
     Generate causal RNN input data. The last target of a segment is used as the 
     target for the whole segment.

    Parameters
    ----------
    data : np.ndarray
        Data. Shape (n_samples,n_features).
    targets : np.ndarray
        Targets. Shape (n_samples,1).
    L : int
        Length of input sequence.
    D : int
        ??

    Returns
    -------
    Tuple[np.ndarray,np.ndarray]. (data,target)
        data: Shape (nsamples,segment_len,n_features)
        target: Shape (nsamples,nclasses)
    """
    W = data.shape[1]; # number of features
    O = targets.shape[1]; # number of classes

    data=np.concatenate((data,targets),axis=1);
    data = pd.DataFrame(data);
    sequence_group=[]
    for i in range(len(data)-L-D):
        sequence_group.append(data.iloc[i:i+L+D])  
    data_= np.array([df.values for df in sequence_group])
    input = data_[:,:L,0:W]
    target = data_[:,-1,W:W+O]
    return input,target

def sequence_generator_non_causal(data:np.ndarray,targets:np.ndarray,segment_len:int)->Tuple[np.ndarray,np.ndarray]:
    """
    Generate Non-causal RNN input data. The first target of a segment is used as 
    the target for the whole segment.

    Parameters
    ----------
    data : np.ndarray
        Data. Shape (n_samples,n_features).
    targets : np.ndarray
        Targets. Shape (n_samples,n_classes).
    segment_len : int
        Segment length.
    nchan : int
        Number of channels.
    nclasses : int
        Number of classes.

    Returns
    -------
    Tuple[np.ndarray,np.ndarray]. (data,target)
        data: Shape (nsamples,segment_len,n_features)
        target: Shape (nsamples,nclasses)


    """
    nchan=data.shape[1];
    nclasses=targets.shape[1];

    data=np.concatenate((data,targets),axis=1);
    data = pd.DataFrame(data);
    sequence_group=[]
    for i in range(len(data)-(segment_len-1)):
        sequence_group.append(data.iloc[i:i+segment_len])  
    data_= np.array([df.values for df in sequence_group])
    inputs = data_[:,:segment_len,0:nchan]
    target = data_[:,0,nchan:nchan+nclasses]

    return inputs,target

def test_sequence_generator_non_causal()->None:
    """
    Test sequence_generator_non_causal
    """
    data=np.array([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6]]);
    targets=np.array([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6]]);
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
    actual_inputs,actual_targets=sequence_generator_non_causal(data,targets, 3);
    assert np.all(expected_inputs==actual_inputs),f"Expected inputs is:\n {expected_inputs} \n but got\n {actual_inputs}";
    assert np.all(expected_targets==actual_targets),f"Expected targets is:\n {expected_targets} \n but got\n {actual_targets}";
    print('Pass')
    
def dict_replace_none_with_empty_array(data):
    """
    Recursively iterates through a Python dictionary and replaces any entry with a value of `None` with an empty array.

    Parameters
    ----------
    data : dict or list or any valid Python data type
        The input data to be processed. It can be a dictionary, a list, or any valid Python data type.

    Returns
    -------
    updated_data : dict or list or any valid Python data type
        The processed data with all `None` values replaced by empty arrays.
    """

    if isinstance(data, dict):
        # Create a copy of the dictionary to avoid modifying the original one
        updated_data = data.copy()
        for key, value in data.items():
            # Recursively call the function on nested dictionaries
            updated_data[key] = dict_replace_none_with_empty_array(value)
        return updated_data
    elif isinstance(data, list):
        # Recursively call the function on list elements
        return [dict_replace_none_with_empty_array(item) for item in data]
    elif data is None:
        # Replace None with an empty array
        return []
    else:
        # Return other data types as is
        return data